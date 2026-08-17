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
        "                            on all-ones input; dumps to <out> instead of full MLEM)\n",
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
        else { print_usage(argv[0]); return 1; }
    }

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

    int Nxz = para.Volumen_num_xz;
    int Ny  = para.Volumen_num_y;
    size_t vol_n = (size_t)Nxz * Nxz * Ny;

    /* ── Initial volume (all ones) ── */
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

    /* ── Run selected mode ── */
    if (!strcmp(mode_str, "cpu")) {
        printf("\n=== CPU mode, %d epochs ===\n", epochs);

        t_start = get_time_sec();
        reconstruct_cpu(proj_measured, volume, &para, epochs);
        t_end = get_time_sec();
        printf("CPU time: %.2f s\n", t_end - t_start);

    } else if (!strcmp(mode_str, "gpu-buf") || !strcmp(mode_str, "gpu-img")) {
        GPUMode gmode = (!strcmp(mode_str, "gpu-img")) ? GPU_MODE_IMAGE : GPU_MODE_BUFFER;
        printf("\n=== GPU mode (%s), %d epochs ===\n", mode_str, epochs);

        CLState cl;
        if (gpu_init(&cl, gmode, kernel_dir) != 0) return 1;

        t_start = get_time_sec();
        reconstruct_gpu(&cl, &para, proj_measured, volume, epochs);
        t_end = get_time_sec();

        printf("GPU time: %.2f s\n", t_end - t_start);
        gpu_cleanup(&cl);

    } else if (!strcmp(mode_str, "gpu-opt")) {
        printf("\n=== GPU-OPT mode (LUT+local+float4+unroll), %d epochs ===\n", epochs);

        CLState cl;
        if (gpu_init(&cl, GPU_MODE_OPT, kernel_dir) != 0) return 1;

        t_start = get_time_sec();
        reconstruct_gpu_opt(&cl, &para, proj_measured, volume, epochs);
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
