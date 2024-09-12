PROGRAM heat_equation
  USE MPI

  IMPLICIT NONE

  INTEGER, PARAMETER :: nx = 100
  INTEGER, PARAMETER :: nt = 100
  REAL, PARAMETER :: L = 1.0
  REAL, PARAMETER :: alpha = 0.1
  REAL, PARAMETER :: t_end = 0.1

  INTEGER :: i, j
  REAL :: dx, dt
  REAL, DIMENSION(nx) :: u, du_dt
  REAL, DIMENSION(nx) :: u_exact

  INTEGER :: ierr
  INTEGER :: num_procs, rank

  CALL MPI_INIT(ierr)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

  dx = L / (nx - 1)
  dt = t_end / nt

  IF (rank == 0) THEN
    DO i = 1, nx
      u(i) = SIN(M_PI * i * dx)
    END DO
  END IF

  CALL MPI_BCAST(u, nx, MPI_REAL, 0, MPI_COMM_WORLD, ierr)

  DO j = 1, nt
    CALL heat_equation_mpi(u, du_dt, dx, alpha, nx)
    CALL euler_step_mpi(u, du_dt, dt, nx)
  END DO

  IF (rank == 0) THEN
    DO i = 1, nx
      u_exact(i) = SIN(M_PI * i * dx) * EXP(-alpha * M_PI * M_PI * t_end)
    END DO

    WRITE (*,*) 'Numerical solution:'
    DO i = 1, nx
      WRITE (*,*) 'u(', i, ') = ', u(i)
    END DO

    WRITE (*,*) 'Exact solution:'
    DO i = 1, nx
      WRITE (*,*) 'u_exact(', i, ') = ', u_exact(i)
    END DO
  END IF

  CALL MPI_FINALIZE(ierr)

CONTAINS

  SUBROUTINE heat_equation_mpi(u, du_dt, dx, alpha, nx)
    REAL, INTENT(IN) :: u(nx)
    REAL, INTENT(OUT) :: du_dt(nx)
    REAL, INTENT(IN) :: dx, alpha
    INTEGER, INTENT(IN) :: nx

    INTEGER :: i

    DO i = 2, nx-1
      du_dt(i) = alpha * (u(i+1) - 2*u(i) + u(i-1)) / dx / dx
    END DO
  END SUBROUTINE heat_equation_mpi

  SUBROUTINE euler_step_mpi(u, du_dt, dt, nx)
    REAL, INTENT(INOUT) :: u(nx)
    REAL, INTENT(IN) :: du_dt(nx)
    REAL, INTENT(IN) :: dt
    INTEGER, INTENT(IN) :: nx

    INTEGER :: i

    DO i = 1, nx
      u(i) = u(i) + dt * du_dt(i)
    END DO
  END SUBROUTINE euler_step_mpi

END PROGRAM heat_equation

