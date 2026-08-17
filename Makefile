CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -ffast-math -Wall -Iinclude -DCL_TARGET_OPENCL_VERSION=120
# Detect OS for OpenCL + HDF5 linking
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    LDFLAGS = -lm -lhdf5 -framework OpenCL
else
    # Ubuntu/Debian installs HDF5 as hdf5_serial; fall back to hdf5 if not found
    HDF5_LIB := $(shell ldconfig -p 2>/dev/null | grep -q libhdf5_serial && echo hdf5_serial || echo hdf5)
    HDF5_INC := $(shell test -d /usr/include/hdf5/serial && echo /usr/include/hdf5/serial || echo /usr/include)
    CFLAGS  += -I$(HDF5_INC)
    LDFLAGS  = -lm -l$(HDF5_LIB) -lOpenCL
endif

SRC_DIR    = src
KERNEL_DIR = kernels
BUILD_DIR  = build

SRCS = $(SRC_DIR)/main.c \
       $(SRC_DIR)/utils.c \
       $(SRC_DIR)/ct_cpu.c \
       $(SRC_DIR)/ct_gpu.c

OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))
TARGET = $(BUILD_DIR)/ct_recon

.PHONY: all clean run-cpu run-gpu-buf run-gpu-img run-gpu-opt \
               run-cpu-512 run-gpu-buf-512 run-gpu-img-512 run-gpu-opt-512 \
               run-python run-op-fp run-op-bp run-op-fp-512 run-op-bp-512

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Run targets ──
DATA256 ?= /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
DATA512 ?= /lgrp/edu-2026-1-gpulab/proj_512_75.hdf5
DATA    ?= $(DATA256)
EPOCHS    ?= 100
SAMPLES   ?= 0      # 0 = auto (Nxz); override e.g. SAMPLES=384
SAMPLES512 ?= 512   # full Nyquist sampling for 512^3
OMP_THREADS ?= $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 8)

# 256 targets (n_samples defaults to Nxz=256)
run-cpu:
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PROC_BIND=close OMP_PLACES=cores $(TARGET) --data $(DATA256) --out output_cpu.hdf5         --mode cpu     --epochs $(EPOCHS) --kernels $(KERNEL_DIR)
run-gpu-buf:
	$(TARGET) --data $(DATA256) --out output_gpu_buf.hdf5     --mode gpu-buf --epochs $(EPOCHS) --kernels $(KERNEL_DIR)
run-gpu-img:
	$(TARGET) --data $(DATA256) --out output_gpu_img.hdf5     --mode gpu-img --epochs $(EPOCHS) --kernels $(KERNEL_DIR)
run-gpu-opt:
	$(TARGET) --data $(DATA256) --out output_gpu_opt.hdf5     --mode gpu-opt --epochs $(EPOCHS) --kernels $(KERNEL_DIR)

# 512 targets (n_samples=384 by default, override with SAMPLES512=N)
run-cpu-512:
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PROC_BIND=close OMP_PLACES=cores $(TARGET) --data $(DATA512) --out output_cpu_512.hdf5     --mode cpu     --epochs $(EPOCHS) --samples $(SAMPLES512) --kernels $(KERNEL_DIR)
run-gpu-buf-512:
	$(TARGET) --data $(DATA512) --out output_gpu_buf_512.hdf5     --mode gpu-buf --epochs $(EPOCHS) --samples $(SAMPLES512) --kernels $(KERNEL_DIR)
run-gpu-img-512:
	$(TARGET) --data $(DATA512) --out output_gpu_img_512.hdf5 --mode gpu-img --epochs $(EPOCHS) --samples $(SAMPLES512) --kernels $(KERNEL_DIR)
run-gpu-opt-512:
	$(TARGET) --data $(DATA512) --out output_gpu_opt_512.hdf5 --mode gpu-opt --epochs $(EPOCHS) --samples $(SAMPLES512) --kernels $(KERNEL_DIR)


# ── Validation helpers ──
# Python reference: must use the same EPOCHS as the C/GPU run being compared
# (validate.py's "MSE vs Python" column is only meaningful when they match).
run-python:
	python3 run_python_reference.py --data $(DATA256) --out output_python.hdf5 --epochs $(EPOCHS)

# Component tests: isolate fp/bp correctness from accumulated MLEM iteration.
# OMP_NUM_THREADS/PROC_BIND/PLACES matches run-cpu/run-cpu-512 — without
# these, fp_cpu still runs OpenMP-parallel (defaults to all cores) but loses
# thread pinning, which matters most on fp's 512^3 workload.
run-op-fp:
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PROC_BIND=close OMP_PLACES=cores $(TARGET) --data $(DATA256) --out fp_cpu.hdf5 --mode cpu --op fp --kernels $(KERNEL_DIR)
run-op-bp:
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PROC_BIND=close OMP_PLACES=cores $(TARGET) --data $(DATA256) --out bp_cpu.hdf5 --mode cpu --op bp --kernels $(KERNEL_DIR)
run-op-fp-512:
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PROC_BIND=close OMP_PLACES=cores $(TARGET) --data $(DATA512) --out fp_cpu_512.hdf5 --mode cpu --op fp --samples $(SAMPLES512) --kernels $(KERNEL_DIR)
run-op-bp-512:
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PROC_BIND=close OMP_PLACES=cores $(TARGET) --data $(DATA512) --out bp_cpu_512.hdf5 --mode cpu --op bp --kernels $(KERNEL_DIR)

clean:
	rm -rf $(BUILD_DIR)
