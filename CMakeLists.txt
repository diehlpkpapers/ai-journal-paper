cmake_minimum_required(VERSION 3.20)

project(my_project VERSION 0.1
    DESCRIPTION "AI generated code"
    LANGUAGES Fortran CXX)

enable_language(Fortran)

set(WITH_CUDA OFF CACHE BOOL "Build CUDA examples")
set(WITH_MPI OFF CACHE BOOL "Build MPI examples")

find_package(OpenMP REQUIRED)

IF(WITH_MPI)
	find_package(MPI REQUIRED)
ENDIF()

set (CMAKE_EXE_LINKER_FLAGS "-lstdc++")


add_subdirectory(fortran)
add_subdirectory(cpp)

IF(WITH_CUDA)
 add_subdirectory(cuda)
 add_subdirectory(cuda_ampere)
ENDIF()

