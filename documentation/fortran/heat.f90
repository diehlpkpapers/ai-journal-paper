PROGRAM heat_equation_1d_parallel
  !==============================================================
  ! Solves the 1D heat equation using parallel computing
  !==============================================================

  USE omp_lib
  IMPLICIT NONE

  !==============================================================
  ! Constants and parameters
  !==============================================================
  INTEGER, PARAMETER :: NX = 100
    ! Number of grid points in x-direction

  INTEGER, PARAMETER :: NT = 1000
    ! Number of time steps

  REAL, PARAMETER :: LX = 1.0
    ! Length of the domain in x-direction

  REAL, PARAMETER :: T_FINAL = 0.1
    ! Final time

  REAL, PARAMETER :: ALPHA = 0.1
    ! Thermal diffusivity

  REAL, PARAMETER :: DX = LX / (NX - 1)
    ! Grid spacing in x-direction

  REAL, PARAMETER :: DT = T_FINAL / NT
    ! Time step

  REAL, Parameter :: M_PI = 4.D0 * DATAN(1.D0)
    ! Mathematical constant pi

  INTEGER :: i, j
    ! Loop counters

  REAL, DIMENSION(NX) :: u, u_new
    ! Arrays to store the solution at the current and next time steps

  !==============================================================
  ! Initialize the grid
  !==============================================================
  DO i = 1, NX
    u(i) = SIN(M_PI * (i - 1) * DX)
      ! Initial condition: sinusoidal function
  END DO

  !==============================================================
  ! Time-stepping loop
  !==============================================================
  DO j = 1, NT
    !$OMP PARALLEL DO PRIVATE(i)
    DO i = 2, NX - 1
      u_new(i) = u(i) + ALPHA * DT / DX / DX * (u(i + 1) - 2 * u(i) + u(i - 1))
        ! Compute the solution at the next time step using the finite difference method
    END DO
    !$OMP END PARALLEL DO

    ! Update the grid
    u = u_new
  END DO

  !==============================================================
  ! Output the final solution
  !==============================================================
  DO i = 1, NX
    WRITE (*,*) u(i)
      ! Print the final solution
  END DO

END PROGRAM heat_equation_1d_parallel

