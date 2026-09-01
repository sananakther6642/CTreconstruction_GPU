#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

/* Scan parameters loaded from HDF5 */
typedef struct {
    double voxelSize;
    double pixelSize;
    double SDD;
    double SOD;
    int    Volumen_num_xz;
    int    Volumen_num_y;
    int    detector_width;
    int    detector_height;
    int    num_projs;
    double *angles;       /* [num_projs] */
    int    n_samples;     /* ray samples per projection (default: Volumen_num_xz) */
    int    use_half_vol;  /* 1 = half-precision vol_img texture (default 0 = float32) */
} CBpara;

/* Load projections + scan parameters from HDF5 */
/* projections: allocated here, caller must free, shape [num_projs * H * W] */
int load_hdf5(const char *path, CBpara *para, float **projections);

/* Save reconstructed volume to HDF5 */
int save_hdf5(const char *path, const CBpara *para, const float *volume);

/* Save a projection stack [num_projs, H, W] to HDF5 as dataset "Projection".
 * Used by --op fp component test to dump fp output for cross-validation. */
int save_hdf5_proj(const char *path, const CBpara *para, const float *proj);

/* Timing utility (seconds) */
double get_time_sec(void);

/* ── FOV mask threshold (see utils.c for the full rationale) ────────────
 * MLEM's update is v *= bp_ratio/bp_ones, historically guarded only by
 * bp_ones > 1e-10f. bp_ones is the sensitivity map: at FOV-edge voxels it
 * is genuinely near zero, so that division multiplies by up to ~1e10 and
 * amplifies any tiny CPU/GPU difference every epoch. Measured at 256^3:
 * MSE(gpu-img, gpu-buf) grows 8.4e-10 -> 1.6e-7 between 10 and 50 epochs
 * (190x for 5x the iterations), driven by a handful of edge voxels, while
 * bypassing the hardware sampler entirely moved it only 6%.
 *
 * Returns the absolute cutoff to use for a given bp_ones array: voxels
 * below it are outside the reliably-sampled region and must be MASKED TO
 * ZERO rather than left frozen (the volume is initialised to 1.0 in
 * main.c, so freezing would leave a bright rim, not a clean edge).
 *
 * Relative to max(bp_ones), not absolute, so the same setting transfers
 * across resolutions and angle counts. rel <= 0 disables masking and
 * returns the legacy 1e-10f, preserving the old behaviour exactly. */
float fov_mask_threshold(const float *bp_ones, size_t n, float rel);

/* Reads FOV_MASK_REL from the environment (0 = off, the default). Both the
 * CPU and GPU paths call this so a single env var drives both and they can
 * never silently disagree. */
float fov_mask_rel_from_env(void);

/* Append one convergence row to path (CSV, header written on first call
 * per path via "w" vs "a" — caller passes epoch==0 to truncate/header).
 * p0, b: measured / current-estimate projections, both [proj_n].
 * v_cur, v_prev: current/previous volume, both [vol_n] (v_prev may be NULL
 * on epoch 0, in which case rel_change is written as 0).
 * Computes and logs: Poisson log-likelihood sum(p0*log(b)-b) (b>0 terms
 * only), data-fidelity residual norm(p0-b)/norm(p0), and relative volume
 * change norm(v_cur-v_prev)/norm(v_cur). */
void log_convergence(const char *path, int epoch, double epoch_time_s,
                      const float *p0, const float *b, size_t proj_n,
                      const float *v_cur, const float *v_prev, size_t vol_n);

/*
 * OSEM: compute the angle permutation that makes each
 * of S subsets both angularly interleaved (subset s gets angles
 * {s, s+S, s+2S, ...}) and visited in maximally-separated order (subsets
 * themselves reordered by a stride coprime to S, close to the golden
 * ratio). After applying this permutation to both para->angles and the
 * projection stack (each angle's [H][W] block, via
 * permute_projections_inplace below), subset k is exactly the contiguous
 * angle range [k*num_projs/S, (k+1)*num_projs/S).
 *
 * perm[i] = original index of the angle that should end up at permuted
 * position i. Caller allocates perm[num_projs]. S==1 fills the identity
 * permutation (perm[i]=i) -- the required no-op path for --subsets 1.
 * num_projs need not be evenly divisible by S.
 */
void compute_osem_permutation(int num_projs, int S, int *perm);

/*
 * Apply perm (as returned by compute_osem_permutation) to a projection
 * stack in place: proj[i] holds a [H*W] block; after this call, the
 * block that was at perm[i] is at i. Also permutes angles[] the same way.
 * Allocates and frees an O(H*W) scratch block internally (not O(proj_n)),
 * via a cycle-following in-place permutation.
 */
void permute_projections_inplace(float *proj, double *angles,
                                  int num_projs, size_t block_elems,
                                  const int *perm);

#endif /* UTILS_H */
