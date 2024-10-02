#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> matmul(const vector<vector<double>>& A, const vector<double>& p) {
    int n = A.size();
    vector<double> Ap(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ap[i] += A[i][j] * p[j];
        }
    }
    return Ap;
}

double dot_product(const vector<double>& a, const vector<double>& b) {
    double res = 0.0;
    for (int i = 0; i < a.size(); i++) {
        res += a[i] * b[i];
    }
    return res;
}

void gaussian_elimination(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < n; k++) {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
    }

    x = vector<double>(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
}

int main() {
    const int n = 3;
    vector<vector<double>> A(n, vector<double>(n, 0.0));
    vector<double> b(n, 0.0), x_cg(n, 0.0), x_ge(n, 0.0), r(n, 0.0), p(n, 0.0), Ap(n, 0.0);
    double tol = 1e-5, rho, rho_new, alpha, beta;
    int max_iter = 100;

    // Define the matrix A and the right-hand side vector b
    A[0][0] = 4; A[0][1] = -1; A[0][2] = 0;
    A[1][0] = -1; A[1][1] = 4; A[1][2] = -1;
    A[2][0] = 0; A[2][1] = -1; A[2][2] = 4;
    b[0] = 1; b[1] = 2; b[2] = 3;

    // Initialize the residual vector r
    r = b;

    // Initialize the search direction vector p
    p = r;

    // Compute the initial residual norm
    rho = dot_product(r, r);

    // Perform the Conjugate Gradient iteration
    for (int i = 1; i <= max_iter; i++) {
        // Compute the matrix-vector product Ap
        Ap = matmul(A, p);

        // Compute the step size alpha
        alpha = rho / dot_product(p, Ap);

        // Update the solution vector x_cg
        for (int j = 0; j < n; j++) {
            x_cg[j] += alpha * p[j];
        }

        // Update the residual vector r
        for (int j = 0; j < n; j++) {
            r[j] -= alpha * Ap[j];
        }

        // Compute the new residual norm
        rho_new = dot_product(r, r);

        // Check for convergence
        if (sqrt(rho_new) < tol) break;

        // Update the search direction vector p
        beta = rho_new / rho;
        for (int j = 0; j < n; j++) {
            p[j] = r[j] + beta * p[j];
        }

        // Update the residual norm
        rho = rho_new;
    }

    // Print the solution using the Conjugate Gradient solver
    cout << "Solution using Conjugate Gradient solver: ";
    for (int i = 0; i < n; i++) {
        cout << x_cg[i] << " ";
    }
    cout << endl;

    // Validate the solution using the Gaussian elimination method
    gaussian_elimination(A, b, x_ge);

    // Print the solution using the Gaussian elimination method
    cout << "Solution
