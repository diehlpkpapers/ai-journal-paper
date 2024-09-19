set (CMAKE_CXX_STANDARD 14)

add_executable(integration_cpp integration.cpp)

add_executable(cg_cpp cg.cpp)

add_executable(heat_shared_cpp heat-shared-fixed.cpp)
target_link_libraries(heat_shared_cpp PRIVATE OpenMP::OpenMP_CXX)
