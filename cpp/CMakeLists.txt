set (CMAKE_CXX_STANDARD 14)

add_executable(integration_cpp integration.cpp)

add_executable(cg_cpp cg.cpp)

add_executable(heat_shared_cpp heat-shared-fixed.cpp)
IF(WITH_OPENMP)
    target_link_libraries(heat_shared_cpp PRIVATE OpenMP::OpenMP_CXX)
ENDIF()

IF(WITH_MPI)
  add_executable(heat_distributed_cpp heat-distributed.cpp)
  target_include_directories(heat_distributed_cpp PRIVATE ${MPI_INCLUDE_PATH})
  target_link_libraries(heat_distributed_cpp ${MPI_CXX_LIBRARIES})
ENDIF()
