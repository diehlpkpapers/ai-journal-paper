#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Function to perform matrix-vector multiplication
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

// Function to perform dot product of two vectors
double dotProduct(vector<double> a, vector<double> b) {
    int n = a.size();
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Conjugate Gradient solver
vector<double> conjugateGradient(vector<vector<double>> A, vector<double> b) {
    
    double tol = 1e-9; 
    int maxIter = 10000;

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
