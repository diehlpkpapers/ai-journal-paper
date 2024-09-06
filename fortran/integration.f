PROGRAM riemann_sum
  IMPLICIT NONE

  INTEGER, PARAMETER :: n = 1000
  REAL, PARAMETER :: a = -3.14159265359, b = 2*3.14159265359/3
  REAL :: dx, x, sum, exact_result
  INTEGER :: i

  dx = (b - a) / n
  sum = 0.0

  DO i = 0, n-1
    x = a + i * dx
    sum = sum + SIN(x) * dx
  END DO

  WRITE (*,*) "The area between -π and 2/3π for sin(x) is: ", sum

  exact_result = -COS(b) + COS(a)
  WRITE (*,*) "The exact area between -π and 2/3π for sin(x) is: ", exact_result

END PROGRAM riemann_sum
