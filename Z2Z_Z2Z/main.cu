#include <cuda/std/complex>

#include "cufftMalloc.h"
#include "cufftManaged.h"
#include "cufftdxMalloc.h"

#include "../Common/cuda_helper.h"

int main( int argc, char **argv ) {

    using in_type  = cuda::std::complex<double>;
    using buf_type = in_type;

    const int fft_size { 2048 };
    const int batches { 32768 };

    TestBench<in_type, buf_type> props( fft_size, batches );

    props.Run( [&props] { cufftMalloc<in_type, buf_type>( props ); } );
    props.Run( [&props] { cufftManaged<in_type, buf_type>( props ); } );

    // <uint SIZE, uint ARCH, uint EPT, uint FPB, class IN_TYPE, class BUF_TYPE, class OUT_TYPE>
    props.Run( [&props] { cufftdxMalloc<fft_size, 750, 16, 1, in_type, buf_type>( props ); } );

    props.VerifyResults( );

    return EXIT_SUCCESS;
}
