#include <iostream>
#include <cmath>

int main() {
    const int n = 1000;
    const double a = -M_PI;
    const double b = 2*M_PI/3;
    double dx = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++) {
        double x = a + i * dx;
        sum += std::sin(x) * dx;
    }

    std::cout << "The area between -π and 2/3π for sin(x) is: " << sum << std::endl;

    double exact_result = -std::cos(b) + std::cos(a);
    std::cout << "The exact area between -π and 2/3π for sin(x) is: " << exact_result << std::endl;

    return 0;
}
