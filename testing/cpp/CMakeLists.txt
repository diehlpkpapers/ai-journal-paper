
add_executable(test_integration numerical_integration.cpp)
target_link_libraries(test_integration GTest::gtest_main)
gtest_discover_tests(test_integration)

add_executable(test_cg cg.cpp)
target_link_libraries(test_cg GTest::gtest_main)
gtest_discover_tests(test_cg)

add_executable(test_heat heat.cpp)
target_link_libraries(test_heat GTest::gtest_main)
gtest_discover_tests(test_heat)
