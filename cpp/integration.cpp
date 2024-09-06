#include <iostream>
#include <cmath>

// define the function sin(x)
double f(double x) {
    return sin(x);
}

int main() {
    // define the limits of integration
    double a = -M_PI;
    double b = 2*M_PI/3;

    // define the number of intervals for the Riemann sum
    int n = 1000;

    // compute the width of each interval
    double dx = (b - a) / n;

    // initialize the sum
    double sum = 0;

    // compute the Riemann sum
    for (int i = 0; i < n; i++) {
        double x = a + i * dx;
        sum += f(x) * dx;
    }

    std::cout << "The area between -π and 2/3π for sin(x) is: " << sum << std::endl;

    // compute the exact value of the integral
    double exact_result = -cos(b) + cos(a);

    std::cout << "The exact area between -π and 2/3π for sin(x) is: " << exact_result << std::endl;

    return 0;
}
