#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils.h"
#include "ct_cpu.h"
#include "ct_gpu.h"

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --data <file.hdf5> --out <out.hdf5> --mode <cpu|gpu-buf|gpu-img|gpu-opt>\n"
        "           [--epochs N]    (default: 100)\n"
        "           [--samples N]   (ray samples per projection, default: volume Nxz)\n"
        "           [--kernels <kernel_dir>]  (default: ../kernels)\n"
        "           [--half]        (use half-precision vol_img texture; default: float32)\n"
        "           [--op fp|bp]    (component test: run a single fp or bp call, CPU only,\n"
        "                            on all-ones input; dumps to <out> instead of full MLEM)\n"
        "           [--log-convergence <file.csv>]  (per-epoch loglik/residual/rel_change;\n"
        "                            off by default, zero extra cost when unset)\n"
        "           [--diag repeat-slab:<angle_offset>:<slab_size>:<n_repeats>[:<realloc_at>]]\n"
        "                           (gpu-buf only; perf-v2 Phase A2/A3 variance diagnostic,\n"
        "                            repeats one fixed fp_buffer angle-slab N times and exits;\n"
        "                            optional realloc_at: reallocate d_vol before that repeat)\n"
        "           [--subsets N]   (gpu-opt only; Ordered Subsets EM, default 1 = plain MLEM,\n"
        "                            provably identical to N unset. N>1 permutes the angle\n"
        "                            stack and requires N | num_projs for even subset sizes;\n"
        "                            1 epoch = 1 full pass over all N subsets, i.e.\n"
        "                            --epochs 20 --subsets 5 does the same total work as\n"
        "                            --epochs 100 --subsets 1)\n",
        prog);
}

int main(int argc, char **argv)
{
    const char *data_path    = NULL;
    const char *out_path     = NULL;
    const char *mode_str     = "gpu-buf";
    const char *kernel_dir   = "../kernels";
    int         epochs       = 100;
    int         n_samples    = 0;  /* 0 = auto (Nxz) */
    int         use_half     = 0;  /* default: float32 vol_img (accurate) */
    const char *op_str       = NULL; /* "fp" or "bp": component test mode */
    const char *conv_log     = NULL; /* --log-convergence path, NULL = off */
    const char *diag_str     = NULL; /* --diag repeat-slab:<off>:<size>:<reps> */
    int         subsets      = 1;    /* --subsets N, default 1 = plain MLEM */

    /* ── Parse args ── */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data")    && i+1<argc) data_path   = argv[++i];
        else if (!strcmp(argv[i], "--out")     && i+1<argc) out_path    = argv[++i];
        else if (!strcmp(argv[i], "--mode")    && i+1<argc) mode_str    = argv[++i];
        else if (!strcmp(argv[i], "--epochs")  && i+1<argc) epochs      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kernels") && i+1<argc) kernel_dir  = argv[++i];
        else if (!strcmp(argv[i], "--samples") && i+1<argc) n_samples   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--half"))                use_half    = 1;
        else if (!strcmp(argv[i], "--op")      && i+1<argc) op_str      = argv[++i];
        else if (!strcmp(argv[i], "--log-convergence") && i+1<argc) conv_log = argv[++i];
        else if (!strcmp(argv[i], "--diag")    && i+1<argc) diag_str    = argv[++i];
        else if (!strcmp(argv[i], "--subsets") && i+1<argc) subsets     = atoi(argv[++i]);
        else { print_usage(argv[0]); return 1; }
    }

    if (subsets < 1) { fprintf(stderr, "--subsets must be >= 1\n"); return 1; }

    if (!data_path || !out_path) { print_usage(argv[0]); return 1; }

    /* ── Load data ── */
    CBpara para;
    float *proj_measured = NULL;
    printf("Loading %s ...\n", data_path);
    if (load_hdf5(data_path, &para, &proj_measured) != 0) return 1;
    para.n_samples    = (n_samples > 0) ? n_samples : para.Volumen_num_xz;
    para.use_half_vol = use_half;
    printf("  vol_img precision: %s\n", use_half ? "half" : "float32");

    printf("  Volume:    %d x %d x %d\n", para.Volumen_num_xz, para.Volumen_num_xz, para.Volumen_num_y);
    printf("  Detector:  %d x %d\n", para.detector_width, para.detector_height);
    printf("  Angles:    %d\n", para.num_projs);
    printf("  SOD/SDD:   %.1f / %.1f mm\n", para.SOD, para.SDD);

    /* perf-v2 Phase C1 (OSEM): permute the angle/projection stack BEFORE
     * build_RT_buffers is ever called (that happens inside gpu_init's
     * callers, later) so R_mats/T_vecs/ang_cs all inherit the permuted
     * order automatically. subsets==1 computes and applies the identity
     * permutation (a no-op memcpy-equivalent, verified as such via
     * compute_osem_permutation's own S<=1 fast path) -- required so the
     * default path is PROVABLY unchanged, not just assumed equivalent. */
    if (subsets > 1 && !strcmp(mode_str, "gpu-opt")) {
        int *perm = (int *)malloc((size_t)para.num_projs * sizeof(int));
        compute_osem_permutation(para.num_projs, subsets, perm);
        size_t block_elems = (size_t)para.detector_height * para.detector_width;
        permute_projections_inplace(proj_measured, para.angles, para.num_projs, block_elems, perm);
        free(perm);
        printf("  OSEM subsets: %d (angle stack permuted)\n", subsets);
    } else if (subsets > 1) {
        fprintf(stderr, "--subsets > 1 is only implemented for --mode gpu-opt\n");
        return 1;
    }

    int Nxz = para.Volumen_num_xz;
    int Ny  = para.Volumen_num_y;
    size_t vol_n = (size_t)Nxz * Nxz * Ny;

    /* ── Initial volume: all-ones ── */
    float *volume = (float *)malloc(vol_n * sizeof(float));
    for (size_t i = 0; i < vol_n; i++) volume[i] = 1.f;

    double t_start, t_end;

    /* ── Component test: single fp or bp call on all-ones input, CPU only ──
     * Isolates operator correctness instead of comparing accumulated MLEM
     * iteration output. Compare against validate_ops.py. */
    if (op_str) {
        if (strcmp(mode_str, "cpu")) {
            fprintf(stderr, "--op is only supported with --mode cpu\n");
            return 1;
        }
        if (!strcmp(op_str, "fp")) {
            size_t proj_n = (size_t)para.num_projs * para.detector_height * para.detector_width;
            float *proj_out = (float *)calloc(proj_n, sizeof(float));
            printf("\n=== --op fp: single fp_cpu(ones) call ===\n");
            t_start = get_time_sec();
            fp_cpu(volume, proj_out, &para);
            t_end = get_time_sec();
            printf("fp_cpu time: %.3f s (n_samples=%d)\n", t_end - t_start, para.n_samples);
            if (save_hdf5_proj(out_path, &para, proj_out) != 0) return 1;
            printf("Saved fp(ones) to %s\n", out_path);
            free(proj_out);
        } else if (!strcmp(op_str, "bp")) {
            size_t proj_n_raw = (size_t)para.num_projs * para.detector_height * para.detector_width;
            float *ones_raw = (float *)malloc(proj_n_raw * sizeof(float));
            for (size_t i = 0; i < proj_n_raw; i++) ones_raw[i] = 1.f;
            printf("\n=== --op bp: single bp_cpu(cone_weight(ones)) call ===\n");
            cone_weight_cpu(ones_raw, &para);
            /* bp_cpu expects preprocessed [np][W][H] layout, same as reconstruct_cpu */
            int W = para.detector_width, H = para.detector_height;
            float *ones_p = (float *)malloc((size_t)para.num_projs * W * H * sizeof(float));
            for (int ip = 0; ip < para.num_projs; ip++) {
                const float *s = ones_raw + (size_t)ip * H * W;
                float       *d = ones_p   + (size_t)ip * W * H;
                for (int iw = 0; iw < W; iw++)
                    for (int ih = 0; ih < H; ih++)
                        d[iw*H+ih] = s[(H-1-ih)*W+iw] / (float)para.voxelSize;
            }
            t_start = get_time_sec();
            bp_cpu(ones_p, volume, &para);
            t_end = get_time_sec();
            printf("bp_cpu time: %.3f s\n", t_end - t_start);
            if (save_hdf5(out_path, &para, volume) != 0) return 1;
            printf("Saved bp(ones) to %s\n", out_path);
            free(ones_raw); free(ones_p);
        } else {
            fprintf(stderr, "Unknown --op: %s (expected fp or bp)\n", op_str);
            return 1;
        }
        free(volume); free(proj_measured); free(para.angles);
        return 0;
    }

    /* ── perf-v2 Phase A2/A3 diagnostic: repeat one fixed fp_buffer
     * angle-slab N times and exit. gpu-buf only (fp_buffer is the kernel
     * showing the variance). Format: repeat-slab:<angle_offset>:<slab_size>:<n_repeats> */
    if (diag_str) {
        if (strncmp(diag_str, "repeat-slab:", 12) != 0) {
            fprintf(stderr, "Unknown --diag: %s (expected repeat-slab:<offset>:<size>:<reps>)\n", diag_str);
            return 1;
        }
        int angle_offset = 0, slab_size = 8, n_repeats = 20, realloc_at = 0;
        int n_parsed = sscanf(diag_str + 12, "%d:%d:%d:%d",
                               &angle_offset, &slab_size, &n_repeats, &realloc_at);
        if (n_parsed != 3 && n_parsed != 4) {
            fprintf(stderr, "Malformed --diag repeat-slab spec: %s\n", diag_str);
            return 1;
        }
        CLState cl;
        if (gpu_init(&cl, GPU_MODE_BUFFER, kernel_dir) != 0) return 1;
        gpu_diag_repeat_slab(&cl, &para, volume, angle_offset, slab_size, n_repeats, realloc_at);
        gpu_cleanup(&cl);
        free(volume); free(proj_measured); free(para.angles);
        return 0;
    }

    /* ── Run selected mode ── */
    if (!strcmp(mode_str, "cpu")) {
        printf("\n=== CPU mode, %d epochs ===\n", epochs);

        t_start = get_time_sec();
        reconstruct_cpu(proj_measured, volume, &para, epochs, conv_log);
        t_end = get_time_sec();
        printf("CPU time: %.2f s\n", t_end - t_start);

    } else if (!strcmp(mode_str, "gpu-buf") || !strcmp(mode_str, "gpu-img")) {
        GPUMode gmode = (!strcmp(mode_str, "gpu-img")) ? GPU_MODE_IMAGE : GPU_MODE_BUFFER;
        printf("\n=== GPU mode (%s), %d epochs ===\n", mode_str, epochs);

        CLState cl;
        if (gpu_init(&cl, gmode, kernel_dir) != 0) return 1;
        if (use_half && !cl.has_fp16) {
            fprintf(stderr, "--half requires cl_khr_fp16, which this device does not support "
                            "(see \"cl_khr_fp16: no\" above). Re-run without --half.\n");
            gpu_cleanup(&cl);
            return 1;
        }

        t_start = get_time_sec();
        reconstruct_gpu(&cl, &para, proj_measured, volume, epochs, conv_log);
        t_end = get_time_sec();

        printf("GPU time: %.2f s\n", t_end - t_start);
        gpu_cleanup(&cl);

    } else if (!strcmp(mode_str, "gpu-opt")) {
        printf("\n=== GPU-OPT mode (LUT+local+float4), %d epochs ===\n", epochs);

        CLState cl;
        if (gpu_init(&cl, GPU_MODE_OPT, kernel_dir) != 0) return 1;
        if (use_half && !cl.has_fp16) {
            fprintf(stderr, "--half requires cl_khr_fp16, which this device does not support "
                            "(see \"cl_khr_fp16: no\" above). Re-run without --half.\n");
            gpu_cleanup(&cl);
            return 1;
        }

        t_start = get_time_sec();
        reconstruct_gpu_opt(&cl, &para, proj_measured, volume, epochs, conv_log, subsets);
        t_end = get_time_sec();

        printf("GPU-opt time: %.2f s\n", t_end - t_start);
        gpu_cleanup(&cl);

    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode_str);
        print_usage(argv[0]);
        return 1;
    }

    /* ── Save output ── */
    printf("Saving to %s ...\n", out_path);
    if (save_hdf5(out_path, &para, volume) != 0) return 1;
    printf("Done.\n");

    free(volume);
    free(proj_measured);
    free(para.angles);
    return 0;
}
