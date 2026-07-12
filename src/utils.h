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
} CBpara;

/* Load projections + scan parameters from HDF5 */
/* projections: allocated here, caller must free, shape [num_projs * H * W] */
int load_hdf5(const char *path, CBpara *para, float **projections);

/* Save reconstructed volume to HDF5 */
int save_hdf5(const char *path, const CBpara *para, const float *volume);

/* Timing utility (seconds) */
double get_time_sec(void);

#endif /* UTILS_H */
