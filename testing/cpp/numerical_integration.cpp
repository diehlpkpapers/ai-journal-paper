#include <gtest/gtest.h>
#include "riemann_sum.h" // assume the code is in a header file called riemann_sum.h

TEST(RiemannSumTest, SimpleTest) {
  // Test the Riemann sum for a simple case
  double a = 0.0;
  double b = M_PI / 2.0;
  int n = 1000;
  double expected_result = 1.0; // exact value of the integral
  double tolerance = 1e-3; // tolerance for the comparison

  double result = riemann_sum(a, b, n);
  EXPECT_NEAR(result, expected_result, tolerance);
}

TEST(RiemannSumTest, NegativeLimitsTest) {
  // Test the Riemann sum for a case with negative limits
  double a = -M_PI / 2.0;
  double b = 0.0;
  int n = 1000;
  double expected_result = -1.0; // exact value of the integral
  double tolerance = 1e-3; // tolerance for the comparison

  double result = riemann_sum(a, b, n);
  EXPECT_NEAR(result, expected_result, tolerance);
}

TEST(RiemannSumTest, LargeNTest) {
  // Test the Riemann sum for a large number of intervals
  double a = 0.0;
  double b = M_PI;
  int n = 100000;
  double expected_result = 2.0; // exact value of the integral
  double tolerance = 1e-6; // tolerance for the comparison

  double result = riemann_sum(a, b, n);
  EXPECT_NEAR(result, expected_result, tolerance);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

