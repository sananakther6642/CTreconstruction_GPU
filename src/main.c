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
        "Usage: %s --data <file.hdf5> --out <out.hdf5> --mode <cpu|gpu-buf|gpu-img>\n"
        "           [--epochs N]  (default: 100)\n"
        "           [--kernels <kernel_dir>]  (default: ../kernels)\n",
        prog);
}

/* Compute MSE between two float arrays of length n */
static double mse(const float *a, const float *b, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return s / (double)n;
}

int main(int argc, char **argv)
{
    const char *data_path    = NULL;
    const char *out_path     = NULL;
    const char *mode_str     = "gpu-buf";
    const char *kernel_dir   = "../kernels";
    int         epochs       = 100;

    /* ── Parse args ── */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data")    && i+1<argc) data_path   = argv[++i];
        else if (!strcmp(argv[i], "--out")     && i+1<argc) out_path    = argv[++i];
        else if (!strcmp(argv[i], "--mode")    && i+1<argc) mode_str    = argv[++i];
        else if (!strcmp(argv[i], "--epochs")  && i+1<argc) epochs      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kernels") && i+1<argc) kernel_dir  = argv[++i];
        else { print_usage(argv[0]); return 1; }
    }

    if (!data_path || !out_path) { print_usage(argv[0]); return 1; }

    /* ── Load data ── */
    CBpara para;
    float *proj_measured = NULL;
    printf("Loading %s ...\n", data_path);
    if (load_hdf5(data_path, &para, &proj_measured) != 0) return 1;

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

    /* ── Run selected mode ── */
    if (!strcmp(mode_str, "cpu")) {
        printf("\n=== CPU mode, %d epochs ===\n", epochs);

        /* cone-weight is applied inside reconstruct_cpu via bp_func,
           but we match Python which applies it in bp_func per-call.
           So we apply once here on a copy and pass the weighted projections. */
        size_t proj_bytes = (size_t)para.num_projs * para.detector_height
                            * para.detector_width * sizeof(float);
        float *proj_w = (float *)malloc(proj_bytes);
        memcpy(proj_w, proj_measured, proj_bytes);
        cone_weight_cpu(proj_w, &para);

        t_start = get_time_sec();
        reconstruct_cpu(proj_w, volume, &para, epochs);
        t_end = get_time_sec();
        free(proj_w);
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
