#include <cufft.h>
#include <cufftXt.h>

#include "../Common/cuda_helper.h"

// Define variables to point at callbacks
__device__ cufftCallbackLoadZ d_loadCallbackPtr   = CB_MulAndScaleInputComplex;
__device__ cufftCallbackStoreZ d_storeCallbackPtr = CB_MulAndScaleOutputComplex;

// cuFFT example using explicit memory copies
template<class IN_TYPE, class BUF_TYPE, class OUT_TYPE = IN_TYPE>
void cufftMalloc( TestBench<IN_TYPE, BUF_TYPE> &tb ) {

    Timer timer;

    // Create device data arrays
    BUF_TYPE *d_bufferData;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_bufferData ), tb.buffer_size ) );

    // Create callback parameters
    cb_inParams<BUF_TYPE> h_inParams;
    h_inParams.scale      = tb.scalar;
    h_inParams.multiplier = tb.multi_data_in;

    // Copy callback parameters to device
    cb_inParams<BUF_TYPE> *d_inParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<BUF_TYPE> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<BUF_TYPE> ), cudaMemcpyHostToDevice ) );

    cb_outParams<OUT_TYPE> h_outParams;
    h_outParams.scale      = tb.scalar;
    h_outParams.multiplier = tb.multi_data_out;

    cb_outParams<OUT_TYPE> *d_outParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outParams ), sizeof( cb_outParams<OUT_TYPE> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_outParams, &h_outParams, sizeof( cb_outParams<OUT_TYPE> ), cudaMemcpyHostToDevice ) );

    CUDA_RT_CALL( cufftPlanMany( &tb.fft_forward,
                                 tb.fft_plan.rank,
                                 tb.fft_plan.n,
                                 tb.fft_plan.inembed,
                                 tb.fft_plan.istride,
                                 tb.fft_plan.idist,
                                 tb.fft_plan.onembed,
                                 tb.fft_plan.ostride,
                                 tb.fft_plan.odist,
                                 CUFFT_Z2Z,
                                 tb.fft_plan.batch ) );

    CUDA_RT_CALL( cufftPlanMany( &tb.fft_inverse,
                                 tb.fft_plan.rank,
                                 tb.fft_plan.n,
                                 tb.fft_plan.inembed,
                                 tb.fft_plan.istride,
                                 tb.fft_plan.idist,
                                 tb.fft_plan.onembed,
                                 tb.fft_plan.ostride,
                                 tb.fft_plan.odist,
                                 CUFFT_Z2Z,
                                 tb.fft_plan.batch ) );

    // Create host callback pointers
    cufftCallbackLoadZ  h_loadCallbackPtr;
    cufftCallbackStoreZ h_storeCallbackPtr;

    // Copy device pointers to host
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_loadCallbackPtr, d_loadCallbackPtr, sizeof( h_loadCallbackPtr ) ) );
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_storeCallbackPtr, d_storeCallbackPtr, sizeof( h_storeCallbackPtr ) ) );

    // Set input callback
    CUDA_RT_CALL( cufftXtSetCallback( tb.fft_inverse,
                                      reinterpret_cast<void **>( &h_loadCallbackPtr ),
                                      CUFFT_CB_LD_COMPLEX_DOUBLE,
                                      reinterpret_cast<void **>( &d_inParams ) ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback( tb.fft_inverse,
                                      reinterpret_cast<void **>( &h_storeCallbackPtr ),
                                      CUFFT_CB_ST_COMPLEX_DOUBLE,
                                      reinterpret_cast<void **>( &d_outParams ) ) );

    // Execute FFT plan
    std::printf( "cufftExecC2C - FFT/IFFT - Malloc\t" );
    timer.startGPUTimer( );

    for ( int i = 0; i < tb.loops; i++ ) {
        CUDA_RT_CALL( cufftExecZ2Z( tb.fft_forward,
                                    reinterpret_cast<cufftDoubleComplex *>( tb.input_data ),
                                    reinterpret_cast<cufftDoubleComplex *>( d_bufferData ),
                                    CUFFT_FORWARD ) );
        CUDA_RT_CALL( cufftExecZ2Z( tb.fft_inverse,
                                    reinterpret_cast<cufftDoubleComplex *>( d_bufferData ),
                                    reinterpret_cast<cufftDoubleComplex *>( tb.cufft_malloc_data ),
                                    CUFFT_INVERSE ) );
    }
    timer.stopAndPrintGPU( tb.loops );

    CUDA_RT_CALL( cudaMemPrefetchAsync( tb.cufft_malloc_data, tb.signal_size, cudaCpuDeviceId, 0 ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( d_bufferData ) );
    CUDA_RT_CALL( cudaFree( d_inParams ) );
    CUDA_RT_CALL( cudaFree( d_outParams ) );
}
