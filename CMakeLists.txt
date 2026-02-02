cmake_minimum_required(VERSION 3.10)

# Project setup
project(fast_cloud LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

# Find Dependencies
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)  # <--- NEW: Find OpenCV

# Includes
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${MPI_C_INCLUDE_PATH})
include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS}) # <--- NEW: Include OpenCV headers

# Library Paths
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Executable
add_executable(fast_cloud main.cpp rgbd_cuda.cu)

# Link Libraries
target_link_libraries(fast_cloud 
    ${PCL_LIBRARIES}
    MPI::MPI_C
    MPI::MPI_CXX
    ${OpenCV_LIBS}  # <--- NEW: Link OpenCV libraries
)
