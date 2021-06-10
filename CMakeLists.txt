#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from NVIDIA CORPORATION is strictly prohibited.
#

cmake_minimum_required(VERSION 3.18)

project(cufftDxSamples)

find_package(CUDAToolkit REQUIRED)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Check if we can use --threads for faster compile
if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11)
  if(${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 2)
    set(CMAKE_CUDA_FLAGS "--threads=0")
  endif()
endif()

if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

if(NOT DEFINED CUB_DIR AND DEFINED ENV{CUB_DIR})
  set(CUB_DIR ENV{CUB_DIR})
endif()

if(NOT DEFINED CUFFTDX_DIR AND DEFINED ENV{CUFFTDX_DIR})
  set(CUFFTDX_DIR ENV{CUFFTDX_DIR})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Found CUB: ${CUB_DIR}")
message(STATUS "Found cuFFTDx: ${CUFFTDX_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(C2C_C2C)
add_subdirectory(Z2Z_Z2Z)
add_subdirectory(R2C_C2R)
add_subdirectory(D2Z_Z2D)
add_subdirectory(C2R_R2C)
add_subdirectory(Z2D_D2Z)
