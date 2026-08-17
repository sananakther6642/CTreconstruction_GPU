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

#endif /* UTILS_H */
