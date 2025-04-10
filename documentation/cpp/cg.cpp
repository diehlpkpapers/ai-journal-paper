/**
 * @file conjugateGradient.cpp
 * @brief Implementation of the Conjugate Gradient solver for linear equation systems
 */

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

/**
 * @brief Performs matrix-vector multiplication
 * @param A Matrix (vector of vectors)
 * @param x Vector
 * @return Resulting vector
 */
vector<double> matVecMult(vector<vector<double>> A, vector<double> x) {
    int n = A.size();
    vector<double> result(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

/**
 * @brief Computes the dot product of two vectors
 * @param a Vector
 * @param b Vector
 * @return Dot product value
 */
double dotProduct(vector<double> a, vector<double> b) {
    int n = a.size();
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief Conjugate Gradient solver for linear equation systems
 * @param A Matrix (vector of vectors)
 * @param b Right-hand side vector
 * @param tol Tolerance for convergence (default: 1e-5)
 * @param maxIter Maximum number of iterations (default: 100)
 * @return Solution vector
 */
vector<double> conjugateGradient(vector<vector<double>> A, vector<double> b, double tol = 1e-5, int maxIter = 100) {
    int n = A.size();
    vector<double> x(n, 0.0);
    vector<double> r = b;
    vector<double> p = r;
    double rho = dotProduct(r, r);

    for (int i = 0; i < maxIter; i++) {
        vector<double> Ap = matVecMult(A, p);
        double alpha = rho / dotProduct(p, Ap);
        for (int j = 0; j < n; j++) {
            x[j] += alpha * p[j];
            r[j] -= alpha * Ap[j];
        }
        double rhoNew = dotProduct(r, r);
        double beta = rhoNew / rho;
        for (int j = 0; j < n; j++) {
            p[j] = r[j] + beta * p[j];
        }
        rho = rhoNew;

        if (sqrt(rho) < tol) {
            break;
        }
    }

    return x;
}

/**
 * @brief Main function
 */
int main() {
    // Define the matrix A and the right-hand side vector b
    vector<vector<double>> A = {{4, -1, 0}, {-1, 4, -1}, {0, -1, 4}};
    vector<double> b = {1, 2, 3};

    // Solve the linear equation system using the Conjugate Gradient solver
    vector<double> x_cg = conjugateGradient(A, b);

    cout << "Solution using Conjugate Gradient solver: ";
    for (int i = 0; i < x_cg.size(); i++) {
        cout << x_cg[i] << " ";
    }
    cout << endl;

    // Validate the solution using the Gaussian elimination method
    vector<vector<double>> A_copy = A;
    vector<double> b_copy = b;
    for (int i = 0; i < A_copy.size(); i++) {
        for (int j = i + 1; j < A_copy.size(); j++) {
            double factor = A_copy[j][i] / A_copy[i][i];
            for (int k = i; k < A_copy.size(); k++) {
                A_copy[j][k] -= factor * A_copy[i][k];
            }
            b_copy[j] -= factor * b_copy[i];
        }
    }
    vector<double> x_ge(A_copy.size(), 0.0);
    for (int i = A_copy.size() - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < A_copy.size(); j++) {
            sum += A_copy[i][j] * x_ge[j];
        }
        x_ge[i] = (b_copy[i] - sum) / A_copy[i][i];
    }

    cout << "Solution using Gaussian elimination method: ";
    for (int i = 0; i < x_ge.size(); i++) {
        cout << x_ge[i] << " ";
    }
    cout << endl;


