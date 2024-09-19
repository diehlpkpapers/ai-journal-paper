add_executable(integration integration.f90)

add_executable(cg cg-fixed.f90)

add_executable(heat_shared heat-shared.f90)
target_link_libraries(heat_shared PRIVATE OpenMP::OpenMP_CXX)

