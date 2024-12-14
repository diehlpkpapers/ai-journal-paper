set (CMAKE_CXX_STANDARD 14)

add_executable(integration_translated integration.cpp)
add_executable(cg_translated cg.cpp)

IF(WITH_MPI)
    add_executable(heat_distributed_translated heat-distributed-fixed.cpp)
    target_include_directories(heat_distributed_translated PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(heat_distributed_translated ${MPI_CXX_LIBRARIES})
ENDIF()
