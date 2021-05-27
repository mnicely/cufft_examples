#pragma once

#include <functional>
#include <random>

#include <cuda.h>
#include <cuda/std/complex>

constexpr int index( int i, int j, int k ) {
    return ( i * j + k );
}

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

// ***************** TIMER *******************
class Timer {

  public:
    // GPU Timer
    void startGPUTimer( ) {
        cudaEventCreate( &startEvent, cudaEventBlockingSync );
        cudaEventRecord( startEvent );
    }  // startGPUTimer

    void stopGPUTimer( ) {
        cudaEventCreate( &stopEvent, cudaEventBlockingSync );
        cudaEventRecord( stopEvent );
        cudaEventSynchronize( stopEvent );
    }  // stopGPUTimer

    void printGPUTime( ) {
        cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent );
        std::printf( "%0.2f ms\n", elapsed_gpu_ms );
    }  // printGPUTime

    void printGPUTime( int const &loops ) {
        cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent );
        std::printf( "%0.2f ms\n", elapsed_gpu_ms / loops );
    }  // printGPUTime

    void stopAndPrintGPU( ) {
        stopGPUTimer( );
        printGPUTime( );
    }

    void stopAndPrintGPU( int const &loops ) {
        stopGPUTimer( );
        printGPUTime( loops );
    }

    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};
};
// ***************** TIMER *******************

// ***************** CALLBACK HELPER FUNCTIONS *******************
template<typename T>
struct cb_inParams {
    T *   multiplier;
    float scale;
};

template<typename T>
struct cb_outParams {
    T *   multiplier;
    float scale;
};

// Complex multiplication
template<typename T>
__device__ T ComplexScale( T const &a, float const &scale ) {
    T c;
    c.x = a.x * scale;
    c.y = a.y * scale;
    return ( c );
}

// Complex multiplication
template<typename T>
__device__ T ComplexMul( T const &a, T const &b ) {
    T c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return ( c );
}

// Input Callback
template<typename T>
__device__ T CB_MulAndScaleInputComplex( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr ) {
    cb_inParams<T> *params = static_cast<cb_inParams<T> *>( callerInfo );
    return ( ComplexScale( ComplexMul( static_cast<T *>( dataIn )[offset], ( params->multiplier )[offset] ),
                           params->scale ) );
}

// Input Callback
template<typename T>
__device__ T CB_MulAndScaleInputReal( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr ) {
    cb_inParams<T> *params = static_cast<cb_inParams<T> *>( callerInfo );
    return ( static_cast<T *>( dataIn )[offset] * ( params->multiplier )[offset] * params->scale );
}

// Output Callback
template<typename T>
__device__ void CB_MulAndScaleOutputComplex( void *dataOut, size_t offset, T element, void *callerInfo, void *sharedPtr ) {
    cb_outParams<T> *params { static_cast<cb_outParams<T> *>( callerInfo ) };

    static_cast<T *>( dataOut )[offset] =
        ComplexScale( ComplexMul( element, ( params->multiplier )[offset] ), params->scale );
}

// Output Callback
template<typename T>
__device__ void CB_MulAndScaleOutputReal( void *dataOut, size_t offset, T element, void *callerInfo, void *sharedPtr ) {
    cb_outParams<T> *params { static_cast<cb_outParams<T> *>( callerInfo ) };

    static_cast<T *>( dataOut )[offset] = element * ( params->multiplier )[offset] * params->scale;
}
// ***************** CALLBACK HELPER FUNCTIONS *******************

// ***************** TestBench *******************
template<class IN_TYPE, class BUF_TYPE, class OUT_TYPE = IN_TYPE>
class TestBench {

  public:
    using SampleRunner = std::function<void( )>;

    struct fft_params {
        int rank;        // 1D FFTs
        int n[1];        // Size of the Fourier transform
        int istride;     // Distance between two successive input elements
        int ostride;     // Distance between two successive output elements
        int idist;       // Distance between input batches
        int odist;       // Distance between output batches
        int inembed[1];  // Input size with pitch (ignored for 1D transforms)
        int onembed[1];  // Output size with pitch (ignored for 1D transforms)
        int batch;       // Number of batched executions
    };

    TestBench( int fft_size, int batches, int loops = 1 ) :
        fft_size( fft_size ),
        batches( batches ),
        signal_size( sizeof( IN_TYPE ) * fft_size * batches ),
        buffer_size( sizeof( BUF_TYPE ) * fft_size * batches ),
        loops( loops ),
        rank( 1 ),
        scalar( 1.7 ),
        tolerance( 1e-3f ),
        fft_plan { rank, { fft_size }, 1, 1, fft_size, fft_size, { 0 }, { 0 }, batches } {

        GetCudaDeviceArch( device, arch );

        CUDA_RT_CALL( cudaMallocManaged( &cufft_malloc_data, signal_size ) );
        CUDA_RT_CALL( cudaMallocManaged( &cufft_managed_data, signal_size ) );
        CUDA_RT_CALL( cudaMallocManaged( &cufftdx_data, signal_size ) );
        CUDA_RT_CALL( cudaMallocManaged( &input_data, signal_size ) );
        CUDA_RT_CALL( cudaMallocManaged( &multi_data_in, buffer_size ) );
        CUDA_RT_CALL( cudaMallocManaged( &multi_data_out, signal_size ) );

        FillData( );
    }

    ~TestBench( ) {
        CUDA_RT_CALL( cudaFree( cufft_malloc_data ) );
        CUDA_RT_CALL( cudaFree( cufft_managed_data ) );
        CUDA_RT_CALL( cudaFree( cufftdx_data ) );
        CUDA_RT_CALL( cudaFree( input_data ) );
        CUDA_RT_CALL( cudaFree( multi_data_in ) );
        CUDA_RT_CALL( cudaFree( multi_data_out ) );
        //     CUDA_RT_CALL( cudaDeviceReset( ) );
    }

    void GetCudaDeviceArch( int &device, int &arch ) {
        int major;
        int minor;
        CUDA_RT_CALL( cudaGetDevice( &device ) );

        CUDA_RT_CALL( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device ) );
        CUDA_RT_CALL( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, device ) );

        arch = major * 100 + minor * 10;
    }

    void FillData( ) {
        using scalar_type = typename IN_TYPE::value_type;

        std::mt19937                                eng;
        std::uniform_real_distribution<scalar_type> dist( -5.0, 5.0 );

        for ( int i = 0; i < ( fft_size * batches ); i++ ) {
            scalar_type temp { dist( eng ) };
            input_data[i] = IN_TYPE( temp, temp );
        }

        // Create multipler input signal
        for ( int i = 0; i < ( fft_size * batches ); i++ ) {
            scalar_type temp { dist( eng ) };
            multi_data_in[i] = BUF_TYPE( temp, temp );
        }

        // Create multipler output signal
        for ( int i = 0; i < ( fft_size * batches ); i++ ) {
            scalar_type temp { dist( eng ) };
            multi_data_out[i] = OUT_TYPE( temp, temp );
        }
    }

    void CopyDataToDevice( ) {
        CUDA_RT_CALL( cudaMemPrefetchAsync( cufft_malloc_data, signal_size, device, NULL ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( cufft_managed_data, signal_size, device, NULL ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( cufftdx_data, signal_size, device, NULL ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( input_data, signal_size, device, NULL ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( multi_data_in, buffer_size, device, NULL ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( multi_data_out, signal_size, device, NULL ) );
    }

    void Run( const SampleRunner &runSample ) {

        CopyDataToDevice( );

        runSample( );

        CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    }

    void VerifyResults( ) {
        printf( "\nCompare results [Malloc/Managed]\n" );
        CompareResults( cufft_malloc_data, cufft_managed_data );

        printf( "\nCompare results [Malloc/Dx]\n" );
        CompareResults( cufft_malloc_data, cufftdx_data );
    }

    void CompareResults( const IN_TYPE *ref, const IN_TYPE *alt ) {
        IN_TYPE relError {};
        int     counter {};

        for ( int i = 0; i < batches; i++ ) {
            for ( int j = 0; j < fft_size; j++ ) {
                size_t idx = index( i, fft_size, j );
                relError   = ( ref[idx] - alt[idx] ) / ref[idx];

                if ( relError.real( ) > tolerance ) {
                    printf( "R - Batch %d: Element %d: %f - %f (%0.7f) > %f\n",
                            i,
                            j,
                            ref[idx].real( ),
                            alt[idx].real( ),
                            relError.real( ),
                            tolerance );
                    counter++;
                }

                if ( relError.imag( ) > tolerance ) {
                    printf( "I - Batch %d: Element %d: %f - %f (%0.7f) > %f\n",
                            i,
                            j,
                            ref[idx].imag( ),
                            alt[idx].imag( ),
                            relError.imag( ),
                            tolerance );
                    counter++;
                }
            }
        }
        if ( !counter ) {
            printf( "All values match!\n\n" );
        }
    }

    const int fft_size;
    const int batches;
    const int signal_size;
    const int buffer_size;
    const int loops;
    const int rank;

    const float scalar;
    const float tolerance;  // Compare cuFFT / cuFFTDx results

    int arch;
    int device;

    // Create cufftHandle
    cufftHandle fft_forward;
    cufftHandle fft_inverse;

    cufftType forward;
    cufftType inverse;

    cufftXtCallbackType load_callback;
    cufftXtCallbackType store_callback;

    IN_TYPE * input_data;

    BUF_TYPE *multi_data_in;
    OUT_TYPE *multi_data_out;

    OUT_TYPE *cufft_malloc_data;
    OUT_TYPE *cufft_managed_data;
    OUT_TYPE *cufftdx_data;
    


    fft_params fft_plan;
};

// Outputs are real
template<>
inline void TestBench<float, cuda::std::complex<float>>::FillData( ) {

    std::mt19937                          eng;
    std::uniform_real_distribution<float> dist( -5.0, 5.0 );

    float temp {};

    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp          = dist( eng );
        input_data[i] = temp;
    }

    // Create multipler input signal
    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp             = dist( eng );
        multi_data_in[i] = cuda::std::complex<float>( temp, temp );
    }

    // Create multipler output signal
    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp              = dist( eng );
        multi_data_out[i] = temp;
    }
}

template<>
inline void TestBench<double, cuda::std::complex<double>>::FillData( ) {

    std::mt19937                           eng;
    std::uniform_real_distribution<double> dist( -5.0, 5.0 );

    double temp {};

    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp          = dist( eng );
        input_data[i] = temp;
    }

    // Create multipler input signal
    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp             = dist( eng );
        multi_data_in[i] = cuda::std::complex<double>( temp, temp );
    }

    // Create multipler output signal
    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp              = dist( eng );
        multi_data_out[i] = temp;
    }
}

// Output are complex
template<>
inline void TestBench<cuda::std::complex<float>, float>::FillData( ) {

    std::mt19937                          eng;
    std::uniform_real_distribution<float> dist( -5.0, 5.0 );

    float temp {};

    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp          = dist( eng );
        input_data[i] = cuda::std::complex<float>( temp, temp );
    }

    // Create multipler input signal
    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp             = dist( eng );
        multi_data_in[i] = temp;
    }

    // Create multipler output signal
    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp              = dist( eng );
        multi_data_out[i] = cuda::std::complex<float>( temp, temp );
    }
}

template<>
inline void TestBench<cuda::std::complex<double>, double>::FillData( ) {

    std::mt19937                           eng;
    std::uniform_real_distribution<double> dist( -5.0, 5.0 );

    double temp {};

    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp          = dist( eng );
        input_data[i] = cuda::std::complex<double>( temp, temp );
    }

    // Create multipler input signal
    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp             = dist( eng );
        multi_data_in[i] = temp;
    }

    // Create multipler output signal
    for ( int i = 0; i < ( fft_size * batches ); i++ ) {
        temp              = dist( eng );
        multi_data_out[i] = cuda::std::complex<double>( temp, temp );
    }
}

// Outputs are real
template<>
inline void TestBench<float, cuda::std::complex<float>>::CompareResults( const float *ref, const float *alt ) {

    float relError {};
    int   counter {};

    for ( int i = 0; i < batches; i++ ) {
        for ( int j = 0; j < fft_size; j++ ) {
            size_t idx = index( i, fft_size, j );
            relError   = ( ref[idx] - alt[idx] ) / ref[idx];

            if ( relError > tolerance ) {
                printf(
                    "R - Batch %d: Element %d: %f - %f (%0.7f) > %f\n", i, j, ref[idx], alt[idx], relError, tolerance );
                counter++;
            }
        }
    }
    if ( !counter ) {
        printf( "All values match!\n\n" );
    }
}

// Outputs are real
template<>
inline void TestBench<double, cuda::std::complex<double>>::CompareResults( const double *ref, const double *alt ) {

    double relError {};
    int    counter {};

    for ( int i = 0; i < batches; i++ ) {
        for ( int j = 0; j < fft_size; j++ ) {
            size_t idx = index( i, fft_size, j );
            relError   = ( ref[idx] - alt[idx] ) / ref[idx];

            if ( relError > tolerance ) {
                printf(
                    "R - Batch %d: Element %d: %f - %f (%0.7f) > %f\n", i, j, ref[idx], alt[idx], relError, tolerance );
                counter++;
            }
        }
    }
    if ( !counter ) {
        printf( "All values match!\n\n" );
    }
}

// Outputs are complex
template<>
inline void TestBench<cuda::std::complex<float>, float>::CompareResults( const cuda::std::complex<float> *ref,
                                                                         const cuda::std::complex<float> *alt ) {

    cuda::std::complex<float> relError {};
    int   counter {};

    for ( int i = 0; i < batches; i++ ) {
        for ( int j = 0; j < (fft_size/2+1); j++ ) {
            size_t idx = index( i, fft_size, j );
            relError   = ( ref[idx] - alt[idx] ) / ref[idx];

            if ( relError.real( ) > tolerance ) {
                printf( "R - Batch %d: Element %d: %f - %f (%0.7f) > %f\n",
                        i,
                        j,
                        ref[idx].real( ),
                        alt[idx].real( ),
                        relError.real( ),
                        tolerance );
                counter++;
            }

            if ( relError.imag( ) > tolerance ) {
                printf( "I - Batch %d: Element %d: %f - %f (%0.7f) > %f\n",
                        i,
                        j,
                        ref[idx].imag( ),
                        alt[idx].imag( ),
                        relError.imag( ),
                        tolerance );
                counter++;
            }
        }
    }
    if ( !counter ) {
        printf( "All values match!\n\n" );
    }
}

// Outputs are complex
template<>
inline void TestBench<cuda::std::complex<double>, double>::CompareResults( const cuda::std::complex<double> *ref,
                                                                           const cuda::std::complex<double> *alt ) {

    cuda::std::complex<double> relError {};
    int    counter {};

    for ( int i = 0; i < batches; i++ ) {
        for ( int j = 0; j < (fft_size/2+1); j++ ) {
            size_t idx = index( i, fft_size, j );
            relError   = ( ref[idx] - alt[idx] ) / ref[idx];

            if ( relError.real( ) > tolerance ) {
                printf( "R - Batch %d: Element %d: %f - %f (%0.7f) > %f\n",
                        i,
                        j,
                        ref[idx].real( ),
                        alt[idx].real( ),
                        relError.real( ),
                        tolerance );
                counter++;
            }

            if ( relError.imag( ) > tolerance ) {
                printf( "I - Batch %d: Element %d: %f - %f (%0.7f) > %f\n",
                        i,
                        j,
                        ref[idx].imag( ),
                        alt[idx].imag( ),
                        relError.imag( ),
                        tolerance );
                counter++;
            }
        }
    }
    if ( !counter ) {
        printf( "All values match!\n\n" );
    }
}
// ***************** TestBench *******************