cmake_minimum_required(VERSION 3.20)

project(my_project VERSION 0.1
    DESCRIPTION "AI generated code"
    LANGUAGES Fortran CXX)

enable_language(Fortran)

set(WITH_CUDA OFF CACHE Bool "Build CUDA code")

set (CMAKE_EXE_LINKER_FLAGS "-lstdc++")


add_subdirectory(fortran)
add_subdirectory(cpp)

IF(WITH_CUDA)
 add_subdirectory(cuda)
ENDIF()

