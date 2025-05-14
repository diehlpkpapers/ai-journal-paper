import math
import numpy as np

def exact_integral(a: float, b: float) -> float:
    """Compute the exact integral of sin(x) from a to b"""
    return -math.cos(b) + math.cos(a)

def riemann_sum(a: float, b: float, n: int = 1_000_000) -> float:
    """
    Compute Riemann sum approximation of sin(x) from a to b
    using the midpoint rule for better accuracy
    
    Args:
        a: lower bound
        b: upper bound
        n: number of rectangles (default: 1,000,000)
    
    Returns:
        Approximate integral value
    """
    dx = (b - a) / n  # Width of each rectangle
    x = np.linspace(a + dx/2, b - dx/2, n)  # Midpoints of all intervals
    return np.sum(np.sin(x) * dx)

def main():
    a = -math.pi
    b = 2/3 * math.pi
    n = 1_000_000  # Number of rectangles
    
    # Compute Riemann sum approximation
    approx_area = riemann_sum(a, b, n)
    
    # Compute exact integral
    exact_area = exact_integral(a, b)
    
    # Calculate errors
    abs_error = abs(approx_area - exact_area)
    rel_error = abs_error / abs(exact_area) * 100
    
    # Print results
    print(f"Calculating area of sin(x) from {a:.6f} to {b:.6f}\n")
    print(f"Riemann sum approximation ({n:,} rectangles): {approx_area:.12f}")
    print(f"Exact integral value: {exact_area:.12f}")
    print(f"Absolute error: {abs_error:.12f}")
    print(f"Relative error: {rel_error:.6f} %")

if __name__ == "__main__":
    main()
