# 
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

cmake_minimum_required(VERSION 3.14.0)

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

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)
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

set(CUDART_LIBRARY cudart)
set(CUFFT_LIBRARY cufft_static)
set(CULIBOS_LIBRARY culibos)

if(NOT WIN32)
  find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(CUFFT_LIBRARY cufft_static ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(CULIBOS_LIBRARY culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

add_subdirectory(C2C_C2C)
add_subdirectory(Z2Z_Z2Z)
add_subdirectory(R2C_C2R)
add_subdirectory(D2Z_Z2D)
add_subdirectory(C2R_R2C)
add_subdirectory(Z2D_D2Z)