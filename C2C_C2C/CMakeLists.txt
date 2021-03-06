#
# Copyright( c ) 2020, NVIDIA CORPORATION.All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto.Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from NVIDIA CORPORATION is strictly prohibited.
#

cmake_minimum_required(VERSION 3.18)

project(C2C_C2C LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME} main.cu cufftdxMalloc.h cufftMalloc.h
                               cufftManaged.h)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../Common ${CUB_DIR} # Load before CTK
          ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUFFTDX_DIR})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cufft_static
                                              CUDA::culibos)
