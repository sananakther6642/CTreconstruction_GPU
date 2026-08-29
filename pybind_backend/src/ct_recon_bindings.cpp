/*
 * pybind11 bindings for the CT reconstruction OpenCL/OpenMP backend.
 *
 * IMPORTANT: no `extern "C"` around the headers below. torch's JIT build
 * (torch.utils.cpp_extension.load) compiles every non-.cu/.sycl source,
 * including the plain-C files utils.c/ct_cpu.c/ct_gpu.c, with the C++
 * compiler and -std=c++17 (verified: only .cu/.sycl get a distinct ninja
 * rule). Those symbols therefore get ITANIUM/C++ mangled names in the
 * built .o files. An `extern "C"` block here would declare them with C
 * linkage instead, and the two would not match at link time. The library
 * sources compile cleanly as C++17 as-is (no VLAs, no restrict, all
 * mallocs already explicitly cast), so plain C++ linkage throughout is
 * both required and sufficient -- do not "fix" this by adding extern "C"
 * back.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"
#include "ct_cpu.h"
#include "ct_gpu.h"

namespace py = pybind11;

namespace {

using FloatArr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using DoubleArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

/* RAII holder for a CBpara whose `angles` buffer we own. CBpara.angles is
 * a raw double* with no ownership semantics in the C API -- every caller
 * of load_hdf5 is expected to free() it themselves. We give it the same
 * treatment here instead of leaking or double-freeing across exceptions. */
struct OwnedPara {
    CBpara para{};
    std::vector<double> angles_storage;

    OwnedPara(const DoubleArr &angles_in,
              double voxelSize, double pixelSize, double SDD, double SOD,
              int Nxz, int Ny, int W, int H, int num_projs,
              int n_samples, bool use_half)
    {
        angles_storage.assign(angles_in.data(), angles_in.data() + angles_in.size());
        para.voxelSize = voxelSize;
        para.pixelSize = pixelSize;
        para.SDD = SDD;
        para.SOD = SOD;
        para.Volumen_num_xz = Nxz;
        para.Volumen_num_y = Ny;
        para.detector_width = W;
        para.detector_height = H;
        para.num_projs = num_projs;
        para.angles = angles_storage.data();
        /* main.c:74 -- n_samples<=0 means "auto", falls back to Nxz.
         * A raw 0 reaches OpenCL kernels as a loop bound (ct_gpu.c
         * fp_buffer/fp_image/bp_opt all take n_samples as an int kernel
         * arg) and silently produces garbage rather than erroring, so
         * this fallback is not optional. */
        para.n_samples = (n_samples > 0) ? n_samples : Nxz;
        para.use_half_vol = use_half ? 1 : 0;
    }
};

/* Common input validation shared by every entry point. Raising here
 * turns a foreseeable misuse into a catchable Python exception instead
 * of a wrong answer or (via CL_CHECK's exit(1) deep in ct_gpu.c) a dead
 * interpreter with no traceback -- ct_gpu.c never returns an error code
 * for GPU-side failures, so the wrapper's job is to keep bad input from
 * ever reaching it in the first place. */
void validate_common(const FloatArr &proj, const DoubleArr &angles,
                      int Nxz, int Ny, int W, int H, int num_projs,
                      int epochs)
{
    if (proj.ndim() != 3)
        throw std::invalid_argument("proj must be a 3D array [num_projs, H, W]");
    if (proj.shape(0) != num_projs || proj.shape(1) != H || proj.shape(2) != W)
        throw std::invalid_argument(
            "proj shape (" + std::to_string(proj.shape(0)) + "," +
            std::to_string(proj.shape(1)) + "," + std::to_string(proj.shape(2)) +
            ") does not match (num_projs=" + std::to_string(num_projs) +
            ", H=" + std::to_string(H) + ", W=" + std::to_string(W) +
            "). Note the layout is [num_projs][H][W], not transposed.");
    if (angles.ndim() != 1 || angles.shape(0) != num_projs)
        throw std::invalid_argument("angles must have shape [num_projs]");
    if (Nxz <= 0 || Ny <= 0 || W <= 0 || H <= 0 || num_projs <= 0)
        throw std::invalid_argument("all geometry dimensions must be positive");
    if (epochs < 1)
        throw std::invalid_argument("epochs must be >= 1");

    /* ct_gpu.c uses `int` for proj_n = np*H*W and vol_n = Nxz*Nxz*Ny
     * (e.g. lines building fp_buffer/bp_opt dispatch sizes). main.c only
     * ever exercised the two shipped datasets (256^3, 512^3); a Python
     * caller can hand in arbitrary geometry, so this overflow guard
     * exists specifically because the wrapper widens the input surface
     * beyond what the CLI ever saw. */
    constexpr long long kIntMax = 2147483647LL;
    long long vol_n = (long long)Nxz * (long long)Nxz * (long long)Ny;
    long long proj_n = (long long)num_projs * (long long)H * (long long)W;
    if (vol_n > kIntMax)
        throw std::invalid_argument("Nxz*Nxz*Ny exceeds INT_MAX");
    if (proj_n > kIntMax)
        throw std::invalid_argument("num_projs*H*W exceeds INT_MAX");
}

/* VRAM pre-flight for the OSEM path. ct_gpu.c never queries device
 * memory anywhere -- reconstruct_gpu_opt allocates one full volume-sized
 * d_bp_ones[s] buffer PER SUBSET on top of the fixed d_vol/d_bp_ratio/
 * proj buffers, and every allocation goes through CL_CHECK, which calls
 * exit(1) on failure. At 512^3 with S subsets that's roughly
 * (3+S)*vol_bytes + 4*proj_bytes -- e.g. S=5 needs ~4.6GB, already over
 * a 4GB card, and the C code's only "error handling" for that is to
 * kill the whole Python process with no traceback. This check exists
 * solely to convert that into a catchable exception before gpu_init
 * ever runs. */
void check_vram_budget(cl_device_id device, int Nxz, int Ny, int W, int H,
                        int num_projs, int subsets)
{
    cl_ulong global_mem = 0, max_alloc = 0;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem),
                     &global_mem, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                     &max_alloc, nullptr);

    size_t vol_bytes = (size_t)Nxz * (size_t)Nxz * (size_t)Ny * sizeof(float);
    size_t proj_bytes = (size_t)num_projs * (size_t)H * (size_t)W * sizeof(float);
    size_t required = (size_t)(3 + subsets) * vol_bytes + 4 * proj_bytes;

    if (max_alloc > 0 && vol_bytes > max_alloc) {
        throw std::runtime_error(
            "a single volume buffer (" + std::to_string(vol_bytes / 1000000) +
            " MB) exceeds this device's CL_DEVICE_MAX_MEM_ALLOC_SIZE (" +
            std::to_string(max_alloc / 1000000) + " MB)");
    }
    if (global_mem > 0 && required > global_mem) {
        throw std::runtime_error(
            "estimated VRAM requirement for subsets=" + std::to_string(subsets) +
            " at this resolution is ~" + std::to_string(required / 1000000) +
            " MB, exceeding device memory (~" +
            std::to_string(global_mem / 1000000) + " MB). Reduce --subsets "
            "or use a smaller volume.");
    }
}

/* Allocates the output volume, fills it 1.0f (main.c:119-120's initial
 * estimate), and returns it. numpy owns this buffer from the moment it
 * is created -- nothing in the C reconstruct call can move or reallocate
 * it, so returning py::array_t here (rather than malloc+copy) is both
 * simpler and safe across the whole call, including if it throws
 * partway through (RVO/py::array_t's own refcounting handles that). */
py::array_t<float> make_initial_volume(int Nxz, int Ny)
{
    py::array_t<float> vol({Nxz, Nxz, Ny});
    std::fill(vol.mutable_data(), vol.mutable_data() + vol.size(), 1.0f);
    return vol;
}

/* ---- reconstruct_cpu ------------------------------------------------ */

py::array_t<float> py_reconstruct_cpu(
    FloatArr proj, DoubleArr angles,
    double voxelSize, double pixelSize, double SDD, double SOD,
    int Nxz, int Ny, int W, int H, int num_projs,
    int epochs, int n_samples, bool use_half,
    const std::string &conv_log)
{
    validate_common(proj, angles, Nxz, Ny, W, H, num_projs, epochs);

    OwnedPara owned(angles, voxelSize, pixelSize, SDD, SOD,
                     Nxz, Ny, W, H, num_projs, n_samples, use_half);

    py::array_t<float> volume = make_initial_volume(Nxz, Ny);

    /* Extract every raw pointer BEFORE releasing the GIL -- py::array_t
     * accessors touch PyObject refcounts and must not be called while
     * the GIL is released. */
    const float *proj_ptr = proj.data();
    float *vol_ptr = volume.mutable_data();
    const char *log_ptr = conv_log.empty() ? nullptr : conv_log.c_str();
    const CBpara *p = &owned.para;
    int ep = epochs;

    {
        py::gil_scoped_release release;
        reconstruct_cpu(proj_ptr, vol_ptr, p, ep, log_ptr);
    }
    return volume;
}

/* ---- reconstruct_gpu_buf / reconstruct_gpu_img ----------------------- */

py::array_t<float> py_reconstruct_gpu_mode(
    GPUMode mode,
    FloatArr proj, DoubleArr angles,
    double voxelSize, double pixelSize, double SDD, double SOD,
    int Nxz, int Ny, int W, int H, int num_projs,
    int epochs, int n_samples, bool use_half,
    const std::string &kernel_dir, const std::string &conv_log)
{
    validate_common(proj, angles, Nxz, Ny, W, H, num_projs, epochs);

    OwnedPara owned(angles, voxelSize, pixelSize, SDD, SOD,
                     Nxz, Ny, W, H, num_projs, n_samples, use_half);

    CLState cl{};
    if (gpu_init(&cl, mode, kernel_dir.c_str()) != 0)
        throw std::runtime_error("gpu_init failed (see stderr for the OpenCL build log)");

    /* main.c:212,231 -- reject --half on a device without cl_khr_fp16
     * before it ever reaches a NULL k_f2h kernel (ct_gpu.c only creates
     * k_f2h when has_fp16 is set). */
    if (use_half && !cl.has_fp16) {
        gpu_cleanup(&cl);
        throw std::invalid_argument(
            "use_half=True requested but this device has no cl_khr_fp16 support");
    }

    py::array_t<float> volume = make_initial_volume(Nxz, Ny);

    const float *proj_ptr = proj.data();
    float *vol_ptr = volume.mutable_data();
    const char *log_ptr = conv_log.empty() ? nullptr : conv_log.c_str();
    const CBpara *p = &owned.para;
    int ep = epochs;

    try {
        py::gil_scoped_release release;
        reconstruct_gpu(&cl, p, proj_ptr, vol_ptr, ep, log_ptr);
    } catch (...) {
        gpu_cleanup(&cl);
        throw;
    }
    gpu_cleanup(&cl);
    return volume;
}

py::array_t<float> py_reconstruct_gpu_buf(
    FloatArr proj, DoubleArr angles,
    double voxelSize, double pixelSize, double SDD, double SOD,
    int Nxz, int Ny, int W, int H, int num_projs,
    int epochs, int n_samples, bool use_half,
    const std::string &kernel_dir, const std::string &conv_log)
{
    return py_reconstruct_gpu_mode(GPU_MODE_BUFFER, proj, angles,
        voxelSize, pixelSize, SDD, SOD, Nxz, Ny, W, H, num_projs,
        epochs, n_samples, use_half, kernel_dir, conv_log);
}

py::array_t<float> py_reconstruct_gpu_img(
    FloatArr proj, DoubleArr angles,
    double voxelSize, double pixelSize, double SDD, double SOD,
    int Nxz, int Ny, int W, int H, int num_projs,
    int epochs, int n_samples, bool use_half,
    const std::string &kernel_dir, const std::string &conv_log)
{
    return py_reconstruct_gpu_mode(GPU_MODE_IMAGE, proj, angles,
        voxelSize, pixelSize, SDD, SOD, Nxz, Ny, W, H, num_projs,
        epochs, n_samples, use_half, kernel_dir, conv_log);
}

/* ---- reconstruct_gpu_opt (OSEM) -------------------------------------- */

py::array_t<float> py_reconstruct_gpu_opt(
    FloatArr proj, DoubleArr angles,
    double voxelSize, double pixelSize, double SDD, double SOD,
    int Nxz, int Ny, int W, int H, int num_projs,
    int epochs, int n_samples, bool use_half,
    const std::string &kernel_dir, const std::string &conv_log,
    int subsets)
{
    validate_common(proj, angles, Nxz, Ny, W, H, num_projs, epochs);

    if (subsets < 1)
        throw std::invalid_argument("subsets must be >= 1");
    /* main.c:99 rejects this outright; ct_gpu.c's reconstruct_gpu_opt
     * would otherwise silently degrade S>np to S=1 internally (after
     * the caller has already permuted for the requested S), which is
     * a different, confusing failure mode -- reject up front instead. */
    if (subsets > num_projs)
        throw std::invalid_argument(
            "subsets (" + std::to_string(subsets) +
            ") cannot exceed num_projs (" + std::to_string(num_projs) + ")");

    OwnedPara owned(angles, voxelSize, pixelSize, SDD, SOD,
                     Nxz, Ny, W, H, num_projs, n_samples, use_half);

    /* Unconditional copy of the projection data. FloatArr's forcecast
     * only copies when the caller's array doesn't already match
     * (float32, C-contiguous) -- for a well-formed array it ALIASES the
     * caller's numpy buffer instead. permute_projections_inplace below
     * mutates its proj argument in place, so relying on forcecast having
     * copied would mean the wrapper mutates the caller's array in the
     * common case and doesn't in the rare one. Always copy here,
     * regardless of what forcecast already did. */
    size_t proj_n = (size_t)num_projs * (size_t)H * (size_t)W;
    std::vector<float> proj_copy(proj.data(), proj.data() + proj_n);

    if (subsets > 1) {
        /* main.c:102-108's OSEM setup, reproduced exactly: compute the
         * permutation, then apply it to BOTH the projection stack and
         * the angles copy already owned by `owned`. ct_gpu.c's own
         * comment on reconstruct_gpu_opt states the per-subset
         * contiguous-range assumption is "valid ONLY because the caller
         * has already permuted" -- this is not optional bookkeeping. */
        std::vector<int> perm(num_projs);
        compute_osem_permutation(num_projs, subsets, perm.data());
        size_t block_elems = (size_t)H * (size_t)W;
        permute_projections_inplace(proj_copy.data(), owned.angles_storage.data(),
                                     num_projs, block_elems, perm.data());
    }

    CLState cl{};
    if (gpu_init(&cl, GPU_MODE_OPT, kernel_dir.c_str()) != 0)
        throw std::runtime_error("gpu_init failed (see stderr for the OpenCL build log)");

    if (use_half && !cl.has_fp16) {
        gpu_cleanup(&cl);
        throw std::invalid_argument(
            "use_half=True requested but this device has no cl_khr_fp16 support");
    }

    try {
        check_vram_budget(cl.device, Nxz, Ny, W, H, num_projs, subsets);
    } catch (...) {
        gpu_cleanup(&cl);
        throw;
    }

    py::array_t<float> volume = make_initial_volume(Nxz, Ny);

    const float *proj_ptr = proj_copy.data();
    float *vol_ptr = volume.mutable_data();
    const char *log_ptr = conv_log.empty() ? nullptr : conv_log.c_str();
    const CBpara *p = &owned.para;
    int ep = epochs;
    int su = subsets;

    try {
        py::gil_scoped_release release;
        reconstruct_gpu_opt(&cl, p, proj_ptr, vol_ptr, ep, log_ptr, su);
    } catch (...) {
        gpu_cleanup(&cl);
        throw;
    }
    gpu_cleanup(&cl);
    return volume;
}

} // namespace

PYBIND11_MODULE(ct_recon, m) {
    m.doc() = "CT cone-beam reconstruction (MLEM/OSEM), OpenCL + OpenMP backend";

    /* Line-buffer C stdout so the library's per-epoch progress printf()s
     * (there is no fflush anywhere in src/*.c) appear as they happen
     * instead of sitting in a 4KB block buffer -- matters most for
     * Jupyter / piped output during a multi-minute run. */
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    m.def("reconstruct_cpu", &py_reconstruct_cpu,
          py::arg("proj"), py::arg("angles"),
          py::arg("voxelSize"), py::arg("pixelSize"), py::arg("SDD"), py::arg("SOD"),
          py::arg("Nxz"), py::arg("Ny"), py::arg("W"), py::arg("H"), py::arg("num_projs"),
          py::arg("epochs") = 100, py::arg("n_samples") = 0, py::arg("use_half") = false,
          py::arg("conv_log") = "",
          "OpenMP CPU reference reconstruction (MLEM). Returns volume [Nxz,Nxz,Ny] float32.");

    m.def("reconstruct_gpu_buf", &py_reconstruct_gpu_buf,
          py::arg("proj"), py::arg("angles"),
          py::arg("voxelSize"), py::arg("pixelSize"), py::arg("SDD"), py::arg("SOD"),
          py::arg("Nxz"), py::arg("Ny"), py::arg("W"), py::arg("H"), py::arg("num_projs"),
          py::arg("epochs") = 100, py::arg("n_samples") = 0, py::arg("use_half") = false,
          py::arg("kernel_dir"), py::arg("conv_log") = "",
          "GPU MLEM reconstruction, manual buffer-gather kernels (--mode gpu-buf).");

    m.def("reconstruct_gpu_img", &py_reconstruct_gpu_img,
          py::arg("proj"), py::arg("angles"),
          py::arg("voxelSize"), py::arg("pixelSize"), py::arg("SDD"), py::arg("SOD"),
          py::arg("Nxz"), py::arg("Ny"), py::arg("W"), py::arg("H"), py::arg("num_projs"),
          py::arg("epochs") = 100, py::arg("n_samples") = 0, py::arg("use_half") = false,
          py::arg("kernel_dir"), py::arg("conv_log") = "",
          "GPU MLEM reconstruction, hardware image-sampler kernels (--mode gpu-img).");

    m.def("reconstruct_gpu_opt", &py_reconstruct_gpu_opt,
          py::arg("proj"), py::arg("angles"),
          py::arg("voxelSize"), py::arg("pixelSize"), py::arg("SDD"), py::arg("SOD"),
          py::arg("Nxz"), py::arg("Ny"), py::arg("W"), py::arg("H"), py::arg("num_projs"),
          py::arg("epochs") = 100, py::arg("n_samples") = 0, py::arg("use_half") = false,
          py::arg("kernel_dir"), py::arg("conv_log") = "", py::arg("subsets") = 1,
          "Optimized GPU reconstruction with Ordered Subsets EM (--mode gpu-opt). "
          "subsets=1 is exactly plain MLEM.");
}
