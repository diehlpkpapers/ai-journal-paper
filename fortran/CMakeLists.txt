add_executable(integration integration.f90)

add_executable(cg cg-fixed.f90)

add_executable(heat_shared heat-shared-fixed.f90)
target_link_libraries(heat_shared PRIVATE OpenMP::OpenMP_CXX)

IF(WITH_MPI)
  add_executable(heat_distributed heat-distributed.f90)
  target_include_directories(heat_distributed PRIVATE ${MPI_INCLUDE_PATH})
  target_link_libraries(heat_distributed PRIVATE MPI::MPI_Fortran)
ENDIF()
