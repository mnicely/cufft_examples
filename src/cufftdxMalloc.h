#include <cufftdx.hpp>

#include "block_io.hpp"
#include "cuda_helper.h"

// cuFFTDx Forward FFT && Inverse FFT CUDA kernel
template<class FFT, class IFFT>
__launch_bounds__( IFFT::max_threads_per_block ) __global__
    void block_fft_ifft_kernel( typename FFT::value_type *               inputData,
                                typename IFFT::value_type *              outputData,
                                cb_inParams<typename FFT::value_type> *  inParams,
                                cb_outParams<typename IFFT::value_type> *outParams ) {

    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    extern __shared__ complex_type shared_mem[];

    // Local array and copy data into it
    complex_type thread_data[FFT::storage_size];
    complex_type temp_mult[FFT::storage_size];
    scalar_type  temp_scale {};

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id { threadIdx.y };

    // Load data from global memory to registers
    example::io<FFT>::load( inputData, thread_data, local_fft_id );
    temp_scale = inParams->scale;

    // Execute FFT
    FFT( ).execute( thread_data, shared_mem );

    example::io<FFT>::load( inParams->multiplier, temp_mult, local_fft_id );

#pragma unroll FFT::elements_per_thread
    for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
        thread_data[i] = ComplexMul( thread_data[i], temp_mult[i] );
        thread_data[i] = ComplexScale( thread_data[i], temp_scale );
    }

    // Execute FFT
    IFFT( ).execute( thread_data, shared_mem );

    example::io<FFT>::load( outParams->multiplier, temp_mult, local_fft_id );
    temp_scale = outParams->scale;

#pragma unroll FFT::elements_per_thread
    for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
        thread_data[i] = ComplexMul( thread_data[i], temp_mult[i] );
        thread_data[i] = ComplexScale( thread_data[i], temp_scale );
    }

    // Save results
    example::io<IFFT>::store( thread_data, outputData, local_fft_id );
}

template<typename T, typename U, uint A, uint SIZE, uint BATCH, uint FPB, uint EPT>
void cufftdxMalloc( const U *inputSignal, const U *multData, const size_t &signalSize, T *h_outputData ) {

    Timer timer;

    // FFT is defined, its: size, type, direction, precision. Block() operator
    // informs that FFT will be executed on block level. Shared memory is
    // required for co-operation between threads.
    using FFT = decltype( cufftdx::Block( ) + cufftdx::Size<SIZE>( ) + cufftdx::Type<cufftdx::fft_type::c2c>( ) +
                          cufftdx::Direction<cufftdx::fft_direction::forward>( ) + cufftdx::Precision<U>( ) +
                          cufftdx::ElementsPerThread<EPT>( ) + cufftdx::FFTsPerBlock<FPB>( ) + cufftdx::SM<A>( ) );

    using IFFT = decltype( cufftdx::Block( ) + cufftdx::Size<SIZE>( ) + cufftdx::Type<cufftdx::fft_type::c2c>( ) +
                           cufftdx::Direction<cufftdx::fft_direction::inverse>( ) + cufftdx::Precision<U>( ) +
                           cufftdx::ElementsPerThread<EPT>( ) + cufftdx::FFTsPerBlock<FPB>( ) + cufftdx::SM<A>( ) );

    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    // Increase dynamic memory limit if required.
    CUDA_RT_CALL( cudaFuncSetAttribute(
        block_fft_ifft_kernel<FFT, IFFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size ) );

    // Copy input data to managed allocation
    complex_type *h_inputData = new complex_type[signalSize];
    for ( int i = 0; i < BATCH * SIZE; i += 2 ) {
        h_inputData[i] = complex_type( inputSignal[i], inputSignal[i + 1] );
    }

    // Create multiplier data
    complex_type *h_multiplier = new complex_type[signalSize];
    for ( int i = 0; i < BATCH * SIZE; i += 2 ) {
        h_multiplier[i] = complex_type { multData[i], multData[i + 1] };
    }

    // Create data arrays and allocate
    complex_type *d_inputData;
    complex_type *d_outputData;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inputData ), signalSize ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outputData ), signalSize ) );

    // Copy input data to device
    CUDA_RT_CALL( cudaMemcpy( d_inputData, h_inputData, signalSize, cudaMemcpyHostToDevice ) );

    complex_type *d_multiplier;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_multiplier ), signalSize ) );
    CUDA_RT_CALL( cudaMemcpy( d_multiplier, h_multiplier, signalSize, cudaMemcpyHostToDevice ) );

    // Create callback parameters
    cb_inParams<complex_type> h_inParams;
    h_inParams.scale      = kScale;
    h_inParams.multiplier = d_multiplier;

    // Copy callback parameters to device
    cb_inParams<complex_type> *d_inParams;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<complex_type> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<complex_type> ), cudaMemcpyHostToDevice ) );

    cb_outParams<complex_type> h_outParams;
    h_outParams.scale      = kScale;
    h_outParams.multiplier = d_multiplier;

    cb_outParams<complex_type> *d_outParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outParams ), sizeof( cb_outParams<complex_type> ) ) );
    CUDA_RT_CALL(
        cudaMemcpy( d_outParams, &h_outParams, sizeof( cb_outParams<complex_type> ), cudaMemcpyHostToDevice ) );

    unsigned int blocks_per_grid { BATCH / FPB };
    // printf("%d: %d: %d: %d: %d\n", blocks_per_grid, FFT::block_dim.x, FFT::block_dim.y, FFT::block_dim.z,
    // FFT::shared_memory_size);

    // Execute FFT plan
    std::printf( "cufftExecC2C - FFT/IFFT - Dx\t\t" );
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
        block_fft_ifft_kernel<FFT, IFFT><<<blocks_per_grid, FFT::block_dim, FFT::shared_memory_size>>>(
            d_inputData, d_outputData, d_inParams, d_outParams );
    }
    timer.stopAndPrintGPU( kLoops );

    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_outputData, signalSize, cudaMemcpyDeviceToHost ) );

    // Cleanup Memory
    delete[]( h_inputData );
    delete[]( h_multiplier );
    CUDA_RT_CALL( cudaFree( d_inputData ) );
    CUDA_RT_CALL( cudaFree( d_outputData ) );
    CUDA_RT_CALL( cudaFree( d_multiplier ) );
    CUDA_RT_CALL( cudaFree( d_inParams ) );
    CUDA_RT_CALL( cudaFree( d_outParams ) );
}