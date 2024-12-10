double f(double x) {
    return sin(x);
}

double riemann_sum(double a,double b, size_t n)
{

     // compute the width of each interval
    double dx = (b - a) / n;

    // initialize the sum
    double sum = 0;

    // compute the Riemann sum
    for (int i = 0; i < n; i++) {
        double x = a + i * dx;
        sum += f(x) * dx;
    }
   
    return sum;


}

