# NOTE
At the moment, C2C examples require https://github.com/mnicely/cub.

# Getting Started
These examples utilize the following toolsets:
* cuFFT
* cuFFTDx (Requires joining CUDA Math Library Early Access Program) https://developer.nvidia.com/CUDAMathLibraryEA
* C++11

# Hardware
Volta+

## cuFFT_vs_cuFFTDx

This code runs three scenarios
1. cuFFT using cudaMalloc
2. cuFFT using cudaMallocManaged
3. cuFFTDx using cudaMalloc

### Objectives
1. Compare coding styles between cuFFT, using cudaMalloc and cudaMallocManaged
2. Compare performance between cuFFT, using cudaMalloc and cudaMallocManaged
3. Compare performance and results between cuFFT and cuFFTDx

### Execution
For float
```bash
mkdir build;
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=75 -DCUB_DIR=${HOME}/workStuff/git_examples/cub -DCUFFTDX_DIR=${HOME}/workStuff/cufft/libcufftdx/include ..
make -j
```

If you don't pass `-DCMAKE_CUDA_ARCHITECTURES=XX` versions CC60, CC70, CC75, and CC80 will be built.

### Output
```bash
$ D2Z_Z2D/D2Z_Z2D 
cufftExecD2Z/Z2D - FFT/IFFT - Managed   29.65 ms
cufftExecD2Z/Z2D - FFT/IFFT - Managed   20.99 ms
cufftExecC2C - FFT/IFFT - Dx            23.31 ms

Compare results [Malloc/Managed]
All values match!

Compare results [Malloc/Dx]
All values match!
```

### Notes
1. This code utilizes cuFFT Callbacks
- https://devblogs.nvidia.com/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
2. This code utilizes separate compilation and linking
- https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/
