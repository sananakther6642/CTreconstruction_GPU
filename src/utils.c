#include "utils.h"
#include <hdf5.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

static float read_float_scalar(hid_t file, const char *name)
{
    float v = 0.f;
    hid_t ds = H5Dopen2(file, name, H5P_DEFAULT);
    if (ds < 0) { fprintf(stderr, "HDF5: cannot open dataset %s\n", name); return 0.f; }
    H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(ds);
    return v;
}

static int read_int_scalar(hid_t file, const char *name)
{
    int v = 0;
    hid_t ds = H5Dopen2(file, name, H5P_DEFAULT);
    if (ds < 0) { fprintf(stderr, "HDF5: cannot open dataset %s\n", name); return 0; }
    H5Dread(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(ds);
    return v;
}

int load_hdf5(const char *path, CBpara *para, float **projections)
{
    hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    para->voxelSize       = (double)read_float_scalar(file, "voxelSize");
    para->pixelSize       = (double)read_float_scalar(file, "pixelSize");
    para->SDD             = (double)read_float_scalar(file, "SDD");
    para->SOD             = (double)read_float_scalar(file, "SOD");
    para->Volumen_num_xz  = read_int_scalar(file, "Volumen_num_xz");
    para->Volumen_num_y   = read_int_scalar(file, "Volumen_num_y");
    para->detector_width  = read_int_scalar(file, "detector_width");
    para->detector_height = read_int_scalar(file, "detector_height");
    para->num_projs       = read_int_scalar(file, "num_projs");

    /* Load angles */
    para->angles = (double *)malloc(para->num_projs * sizeof(double));
    {
        hid_t ds = H5Dopen2(file, "Angle", H5P_DEFAULT);
        H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, para->angles);
        H5Dclose(ds);
    }

    /* Load projections [num_projs, H, W] */
    {
        size_t n = (size_t)para->num_projs *
                   (size_t)para->detector_height *
                   (size_t)para->detector_width;
        *projections = (float *)malloc(n * sizeof(float));
        hid_t ds = H5Dopen2(file, "Projection", H5P_DEFAULT);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *projections);
        H5Dclose(ds);
    }

    H5Fclose(file);
    return 0;
}

int save_hdf5(const char *path, const CBpara *para, const float *volume)
{
    hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "Cannot create %s\n", path); return -1; }

    /* voxelSize scalar */
    {
        hsize_t dims[1] = {1};
        hid_t sp = H5Screate_simple(1, dims, NULL);
        float v = (float)para->voxelSize;
        hid_t ds = H5Dcreate2(file, "voxelSize", H5T_NATIVE_FLOAT, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
        H5Dclose(ds); H5Sclose(sp);
    }

    /* Volume [Nxz, Nxz, Ny] */
    {
        hsize_t dims[3] = {(hsize_t)para->Volumen_num_xz,
                           (hsize_t)para->Volumen_num_xz,
                           (hsize_t)para->Volumen_num_y};
        hid_t sp = H5Screate_simple(3, dims, NULL);
        hid_t ds = H5Dcreate2(file, "Volume", H5T_NATIVE_FLOAT, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, volume);
        H5Dclose(ds); H5Sclose(sp);
    }

    H5Fclose(file);
    return 0;
}

int save_hdf5_proj(const char *path, const CBpara *para, const float *proj)
{
    hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "Cannot create %s\n", path); return -1; }

    hsize_t dims[3] = {(hsize_t)para->num_projs,
                        (hsize_t)para->detector_height,
                        (hsize_t)para->detector_width};
    hid_t sp = H5Screate_simple(3, dims, NULL);
    hid_t ds = H5Dcreate2(file, "Projection", H5T_NATIVE_FLOAT, sp,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, proj);
    H5Dclose(ds); H5Sclose(sp);

    H5Fclose(file);
    return 0;
}

double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* gcd via Euclid's algorithm */
static int gcd_i(int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; }

void compute_osem_permutation(int num_projs, int S, int *perm)
{
    if (S <= 1) {
        for (int i = 0; i < num_projs; i++) perm[i] = i;
        return;
    }

    /* Step B: subset visit order -- stride coprime to S, chosen close to
     * the golden-ratio point S*0.382 so consecutive visited subsets are
     * maximally angularly separated (matches the plan's worked examples:
     * S=5 -> stride 2 -> order 0,2,4,1,3). */
    int best_stride = 1;
    double target = (double)S * 0.381966011; /* 1/phi^2 */
    double best_dist = 1e18;
    for (int g = 1; g < S; g++) {
        if (gcd_i(g, S) != 1) continue;
        double d = (g > target) ? (g - target) : (target - g);
        if (d < best_dist) { best_dist = d; best_stride = g; }
    }

    /* visit_order[v] = subset index visited v-th */
    int *visit_order = (int *)malloc(S * sizeof(int));
    for (int v = 0; v < S; v++) visit_order[v] = (v * best_stride) % S;

    /* Step A: interleave -- subset s (0-indexed, pre-visit-reorder) gets
     * angles {s, s+S, s+2S, ...} from the ORIGINAL angle order. Groups
     * may have different sizes when S doesn't divide num_projs evenly
     * (not the case for this project's 75 angles with S in {3,5,15,25},
     * but handled correctly regardless). */
    int pos = 0;
    for (int v = 0; v < S; v++) {
        int s = visit_order[v];
        for (int orig = s; orig < num_projs; orig += S)
            perm[pos++] = orig;
    }
    free(visit_order);
}

void permute_projections_inplace(float *proj, double *angles,
                                  int num_projs, size_t block_elems,
                                  const int *perm)
{
    /* perm[i] = original index that should end up at position i.
     * In-place permutation via cycle-following: 'done' tracks which
     * destination slots have already received their final value.
     * Only needs O(block_elems) scratch (one block), not O(proj_n). */
    float *scratch = (float *)malloc(block_elems * sizeof(float));
    char *done = (char *)calloc((size_t)num_projs, 1);

    for (int start = 0; start < num_projs; start++) {
        if (done[start]) continue;
        /* Follow the cycle starting at 'start': repeatedly pull the
         * block that belongs at the current slot from perm[cur], until
         * the cycle closes back to 'start'. */
        int cur = start;
        memcpy(scratch, proj + (size_t)cur * block_elems, block_elems * sizeof(float));
        double angle_scratch = angles[cur];
        while (1) {
            int src = perm[cur];
            done[cur] = 1;
            if (src == start) break;
            memcpy(proj + (size_t)cur * block_elems,
                   proj + (size_t)src * block_elems,
                   block_elems * sizeof(float));
            angles[cur] = angles[src];
            cur = src;
        }
        memcpy(proj + (size_t)cur * block_elems, scratch, block_elems * sizeof(float));
        angles[cur] = angle_scratch;
    }

    free(scratch);
    free(done);
}

void log_convergence(const char *path, int epoch, double epoch_time_s,
                      const float *p0, const float *b, size_t proj_n,
                      const float *v_cur, const float *v_prev, size_t vol_n)
{
    double loglik = 0.0;
    double res_num = 0.0, res_den = 0.0;
    for (size_t i = 0; i < proj_n; i++) {
        double p0i = (double)p0[i], bi = (double)b[i];
        if (bi > 1e-12) loglik += p0i * log(bi) - bi;
        double d = p0i - bi;
        res_num += d * d;
        res_den += p0i * p0i;
    }
    double residual = (res_den > 0.0) ? sqrt(res_num / res_den) : 0.0;

    double rel_change = 0.0;
    if (v_prev) {
        double dnum = 0.0, dden = 0.0;
        for (size_t i = 0; i < vol_n; i++) {
            double d = (double)v_cur[i] - (double)v_prev[i];
            dnum += d * d;
            dden += (double)v_cur[i] * (double)v_cur[i];
        }
        rel_change = (dden > 0.0) ? sqrt(dnum / dden) : 0.0;
    }

    FILE *f = fopen(path, (epoch == 0) ? "w" : "a");
    if (!f) { fprintf(stderr, "log_convergence: cannot open %s\n", path); return; }
    if (epoch == 0) fprintf(f, "epoch,time_s,loglik,residual,rel_change\n");
    fprintf(f, "%d,%.6f,%.10g,%.10g,%.10g\n",
            epoch + 1, epoch_time_s, loglik, residual, rel_change);
    fclose(f);
}
