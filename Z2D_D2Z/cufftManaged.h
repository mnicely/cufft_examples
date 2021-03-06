#include <cufft.h>
#include <cufftXt.h>

#include "../Common/cuda_helper.h"

// Define variables to point at callbacks
__device__ __managed__ cufftCallbackLoadD d_loadManagedCallbackPtr   = CB_MulAndScaleInputReal;
__device__ __managed__ cufftCallbackStoreZ d_storeManagedCallbackPtr = CB_MulAndScaleOutputComplex;

template<class IN_TYPE, class BUF_TYPE, class OUT_TYPE = IN_TYPE>
void cufftManaged( TestBench<IN_TYPE, BUF_TYPE> &tb ) {

    Timer timer;

    // Create data arrays
    BUF_TYPE *bufferData;
    CUDA_RT_CALL( cudaMallocManaged( &bufferData, tb.buffer_size ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( bufferData, tb.buffer_size, tb.device, NULL ) );

    // Create callback parameters
    cb_inParams<BUF_TYPE> *inParams;

    CUDA_RT_CALL( cudaMallocManaged( &inParams, sizeof( cb_inParams<BUF_TYPE> ) ) );
    inParams->scale      = tb.scalar;
    inParams->multiplier = tb.multi_data_in;

    cb_outParams<OUT_TYPE> *outParams;

    CUDA_RT_CALL( cudaMallocManaged( &outParams, sizeof( cb_outParams<OUT_TYPE> ) ) );
    outParams->scale      = tb.scalar;
    outParams->multiplier = tb.multi_data_out;

    CUDA_RT_CALL( cudaMemPrefetchAsync( inParams, sizeof( cb_inParams<BUF_TYPE> ), tb.device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outParams, sizeof( cb_outParams<OUT_TYPE> ), tb.device, NULL ) );

    CUDA_RT_CALL( cufftPlanMany( &tb.fft_inverse,
                                 tb.fft_plan.rank,
                                 tb.fft_plan.n,
                                 tb.fft_plan.inembed,
                                 tb.fft_plan.istride,
                                 tb.fft_plan.idist,
                                 tb.fft_plan.onembed,
                                 tb.fft_plan.ostride,
                                 tb.fft_plan.odist,
                                 CUFFT_Z2D,
                                 tb.fft_plan.batch ) );

    CUDA_RT_CALL( cufftPlanMany( &tb.fft_forward,
                                 tb.fft_plan.rank,
                                 tb.fft_plan.n,
                                 tb.fft_plan.inembed,
                                 tb.fft_plan.istride,
                                 tb.fft_plan.idist,
                                 tb.fft_plan.onembed,
                                 tb.fft_plan.ostride,
                                 tb.fft_plan.odist,
                                 CUFFT_D2Z,
                                 tb.fft_plan.batch ) );

    // Set input callback
    CUDA_RT_CALL( cufftXtSetCallback( tb.fft_forward,
                                      reinterpret_cast<void **>( &d_loadManagedCallbackPtr ),
                                      CUFFT_CB_LD_REAL_DOUBLE,
                                      reinterpret_cast<void **>( &inParams ) ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback( tb.fft_forward,
                                      reinterpret_cast<void **>( &d_storeManagedCallbackPtr ),
                                      CUFFT_CB_ST_COMPLEX_DOUBLE,
                                      reinterpret_cast<void **>( &outParams ) ) );

    // Execute FFT plan
    std::printf( "cufftExecZ2D/D2Z - FFT/IFFT - Managed\t" );

    timer.startGPUTimer( );

    for ( int i = 0; i < tb.loops; i++ ) {
        CUDA_RT_CALL( cufftExecZ2D( tb.fft_inverse,
                                    reinterpret_cast<cufftDoubleComplex *>( tb.input_data ),
                                    reinterpret_cast<cufftDoubleReal *>( bufferData ) ) );
        CUDA_RT_CALL( cufftExecD2Z( tb.fft_forward,
                                    reinterpret_cast<cufftDoubleReal *>( bufferData ),
                                    reinterpret_cast<cufftDoubleComplex *>( tb.cufft_managed_data ) ) );
    }
    timer.stopAndPrintGPU( tb.loops );

    CUDA_RT_CALL( cudaMemPrefetchAsync( tb.cufft_managed_data, tb.signal_size, cudaCpuDeviceId, 0 ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( bufferData ) );
    CUDA_RT_CALL( cudaFree( inParams ) );
    CUDA_RT_CALL( cudaFree( outParams ) );
}
