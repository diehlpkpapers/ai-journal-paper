#include <iostream>
#include <cmath>
#include <iomanip>

constexpr double PI = 3.14159265358979323846;

// Function to compute the exact integral of sin(x) from a to b
double exact_integral(double a, double b) {
    return -cos(b) + cos(a);
}

// Function to compute Riemann sum approximation of sin(x) from a to b
double riemann_sum(double a, double b, int n) {
    double dx = (b - a) / n;  // Width of each rectangle
    double sum = 0.0;
    
    for (int i = 0; i < n; ++i) {
        double x = a + (i + 0.5) * dx;  // Midpoint of the interval
        sum += sin(x) * dx;
    }
    
    return sum;
}

int main() {
    double a = -PI;
    double b = 2.0/3.0 * PI;
    int n = 1000000;  // Number of rectangles
    
    // Compute Riemann sum approximation
    double approx_area = riemann_sum(a, b, n);
    
    // Compute exact integral
    double exact_area = exact_integral(a, b);
    
    // Calculate the error
    double error = fabs(approx_area - exact_area);
    
    // Output results
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Calculating area of sin(x) from " << a << " to " << b << "\n\n";
    std::cout << "Riemann sum approximation (" << n << " rectangles): " << approx_area << "\n";
    std::cout << "Exact integral value: " << exact_area << "\n";
    std::cout << "Absolute error: " << error << "\n";
    std::cout << "Relative error: " << (error / fabs(exact_area)) * 100 << " %\n";
    
    return 0;
}
