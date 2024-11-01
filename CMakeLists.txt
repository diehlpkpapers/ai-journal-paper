cmake_minimum_required(VERSION 3.20)

project(my_project VERSION 0.1
    DESCRIPTION "AI generated code"
    LANGUAGES Fortran CXX)

enable_language(Fortran)

set(WITH_CUDA OFF CACHE BOOL "Build CUDA examples")
set(WITH_MPI OFF CACHE BOOL "Build MPI examples")

if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

IF(WITH_MPI)
  find_package(MPI COMPONENTS Fortran CXX REQUIRED)
ENDIF()

set (CMAKE_EXE_LINKER_FLAGS "-lstdc++")


add_subdirectory(fortran)
add_subdirectory(cpp)
add_subdirectory(translation)

IF(WITH_CUDA)
 add_subdirectory(cuda)
 add_subdirectory(cuda_ampere)
ENDIF()

