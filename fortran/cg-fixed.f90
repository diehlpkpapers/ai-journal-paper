PROGRAM conjugate_gradient
  IMPLICIT NONE

  INTEGER, PARAMETER :: n = 3
  REAL, DIMENSION(n,n) :: A
  REAL, DIMENSION(n) :: b, x_cg, x_ge, r, p, Ap
  REAL :: tol = 1e-5, rho, rho_new, alpha, beta
  INTEGER :: i, j, k, max_iter = 100

  ! Define the matrix A and the right-hand side vector b
  A(1,1) = 4; A(1,2) = -1; A(1,3) = 0
  A(2,1) = -1; A(2,2) = 4; A(2,3) = -1
  A(3,1) = 0; A(3,2) = -1; A(3,3) = 4
  b(1) = 1; b(2) = 2; b(3) = 3

  ! Initialize the solution vector x_cg
  x_cg = 0.0

  ! Initialize the residual vector r
  r = b 

  ! Initialize the search direction vector p
  p = r

  ! Compute the initial residual norm
  rho = DOT_PRODUCT(r, r)

  ! Perform the Conjugate Gradient iteration
  DO i = 1, max_iter
    ! Compute the matrix-vector product Ap
    Ap = MATMUL(A, p)

    ! Compute the step size alpha
    alpha = rho / DOT_PRODUCT(p, Ap)
    
    ! Update the solution vector x_cg
    x_cg = x_cg + alpha * p

    ! Update the residual vector r
    r = r - alpha * Ap

    ! Compute the new residual norm
    rho_new = DOT_PRODUCT(r, r)

    ! Check for convergence
    IF (SQRT(rho_new) < tol) EXIT

    ! Update the search direction vector p
    beta = rho_new / rho
    p = r + beta * p

    ! Update the residual norm
    rho = rho_new
  END DO

  ! Print the solution using the Conjugate Gradient solver
  WRITE (*,*) "Solution using Conjugate Gradient solver: "
  WRITE (*,*) x_cg

  ! Validate the solution using the Gaussian elimination method
  CALL gaussian_elimination(A, b, x_ge)

  ! Print the solution using the Gaussian elimination method
  WRITE (*,*) "Solution using Gaussian elimination method: "
  WRITE (*,*) x_ge

  ! Check if the solutions are equal
  IF (ALL(ABS(x_cg - x_ge) < tol)) THEN
    WRITE (*,*) "The solutions are equal!"
  ELSE
    WRITE (*,*) "The solutions are not equal!"
  END IF

CONTAINS

  FUNCTION DOT_PRODUCT(a, b) RESULT(res)
    REAL, DIMENSION(:), INTENT(IN) :: a, b
    REAL :: res
    res = SUM(a * b)
  END FUNCTION DOT_PRODUCT

  SUBROUTINE gaussian_elimination(A, b, x)
    REAL, DIMENSION(:,:), INTENT(OUT) :: A
    REAL, DIMENSION(:), INTENT(OUT) :: b
    REAL, DIMENSION(:), INTENT(OUT) :: x
    INTEGER :: i, j, k
    REAL :: factor

    DO i = 1, SIZE(A,1)
      DO j = i + 1, SIZE(A,1)
        factor = A(j,i) / A(i,i)
        A(j,i:) = A(j,i:) - factor * A(i,i:)
        b(j) = b(j) - factor * b(i)
      END DO
    END DO

    x = 0.0
    DO i = SIZE(A,1), 1, -1
      x(i) = (b(i) - SUM(A(i,i+1:)*x(i+1:))) / A(i,i)
    END DO
  END SUBROUTINE gaussian_elimination

END PROGRAM conjugate_gradient

