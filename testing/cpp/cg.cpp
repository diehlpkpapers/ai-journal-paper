#include <gtest/gtest.h>
#include "conjugate_gradient.h" // assume the code is in a header file called conjugate_gradient.h

TEST(ConjugateGradientTest, SimpleTest) {
  // Define a simple matrix A and right-hand side vector b
  vector<vector<double>> A = {{4, -1, 0}, {-1, 4, -1}, {0, -1, 4}};
  vector<double> b = {1, 2, 3};

  // Solve the linear equation system using the Conjugate Gradient solver
  vector<double> x_cg = conjugateGradient(A, b);
  
  // Define the expected solution
  //vector<double> x_expected = {0.5, 1.0, 1.5};
  vector<double> x_expected = {0.46428571, 0.85714286, 0.96428571};

  // Check if the solution is correct
  for (int i = 0; i < x_cg.size(); i++) {
    EXPECT_NEAR(x_cg[i], x_expected[i], 1e-5);
  }
}

TEST(ConjugateGradientTest, LargeMatrixTest) {
  // Define a larger matrix A and right-hand side vector b
  int n = 10;
  vector<vector<double>> A(n, vector<double>(n, 0.0));
  vector<double> b(n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = (i == j) ? 4.0 : (i == j - 1 || i == j + 1) ? -1.0 : 0.0;
    }
    b[i] = i + 1.0;
  }

  // Solve the linear equation system using the Conjugate Gradient solver
  vector<double> x_cg = conjugateGradient(A, b);

  // Define the expected solution
  vector<double> x_expected = {0.49999026, 0.99996104, 1.49985391, 1.9994546 , 2.49796447,
       2.9924033 , 3.47164873, 3.89419162, 4.10511777, 3.52627944};
  //for (int i = 0; i < n; i++) {
  //  x_expected[i] = (i + 1.0) / 4.0;
  //}

  // Check if the solution is correct
  for (int i = 0; i < x_cg.size(); i++) {
    EXPECT_NEAR(x_cg[i], x_expected[i], 1e-5);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

