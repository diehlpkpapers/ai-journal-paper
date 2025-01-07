cmake_minimum_required(VERSION 3.20)

project(my_project VERSION 0.1
    DESCRIPTION "AI generated code"
    LANGUAGES Fortran CXX)

enable_language(Fortran)
enable_testing()

set(WITH_CUDA OFF CACHE BOOL "Build CUDA examples")
set(WITH_MPI OFF CACHE BOOL "Build MPI examples")
set(WITH_OPENMP OFF CACHE BOOL "Build examples with OpenMP")
set(WITH_UNIT_TESTING OFF CACHE BOOL "Enable unit tests")



set(CMAKE_CXX_FLAGS "-g -O0 -Wall -fprofile-arcs -ftest-coverage")

if(WITH_UNIT_TESTING)
    find_package(GTest)
endif()


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

IF(WITH_UNIT_TESTING)
    add_subdirectory(testing)
    include(GoogleTest)
ENDIF()
