CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -Wall -Iinclude
LDFLAGS = -lm -lhdf5

# Detect OS for OpenCL linking
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    LDFLAGS += -framework OpenCL
else
    LDFLAGS += -lOpenCL
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

.PHONY: all clean run-cpu run-gpu-buf run-gpu-img run-gpu-opt

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Example run targets (set DATA and OUT) ──
DATA    ?= /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5
OUT_CPU ?= output_cpu.hdf5
OUT_BUF ?= output_gpu_buf.hdf5
OUT_IMG ?= output_gpu_img.hdf5
EPOCHS  ?= 100

run-cpu:
	$(TARGET) --data $(DATA) --out $(OUT_CPU) --mode cpu --epochs $(EPOCHS) \
	          --kernels $(KERNEL_DIR)

run-gpu-buf:
	$(TARGET) --data $(DATA) --out $(OUT_BUF) --mode gpu-buf --epochs $(EPOCHS) \
	          --kernels $(KERNEL_DIR)

run-gpu-img:
	$(TARGET) --data $(DATA) --out $(OUT_IMG) --mode gpu-img --epochs $(EPOCHS) \
	          --kernels $(KERNEL_DIR)

OUT_OPT ?= output_gpu_opt.hdf5
run-gpu-opt:
	$(TARGET) --data $(DATA) --out $(OUT_OPT) --mode gpu-opt --epochs $(EPOCHS) \
	          --kernels $(KERNEL_DIR)

clean:
	rm -rf $(BUILD_DIR)
