program riemann_sin_area
    implicit none
    
    ! Declare variables
    real(8), parameter :: pi = 4.0_8 * atan(1.0_8)
    real(8) :: a, b, dx, x, approx_area, exact_area, abs_error, rel_error
    integer :: i, n
    
    ! Integration bounds
    a = -pi
    b = (2.0_8/3.0_8) * pi
    
    ! Number of rectangles (increase for better accuracy)
    n = 1000000
    
    ! Compute Riemann sum approximation (midpoint rule)
    dx = (b - a) / real(n, 8)
    approx_area = 0.0_8
    
    do i = 0, n-1
        x = a + (real(i, 8) + 0.5_8) * dx  ! Midpoint of interval
        approx_area = approx_area + sin(x) * dx
    end do
    
    ! Compute exact integral value
    exact_area = -cos(b) + cos(a)
    
    ! Calculate errors
    abs_error = abs(approx_area - exact_area)
    rel_error = (abs_error / abs(exact_area)) * 100.0_8
    
    ! Display results
    write(*, '(a,f10.6,a,f10.6)') 'Calculating area of sin(x) from ', a, ' to ', b
    write(*, '(a)') ''
    write(*, '(a,i0,a,f20.12)') 'Riemann sum approximation (', n, ' rectangles): ', approx_area
    write(*, '(a,f20.12)') 'Exact integral value: ', exact_area
    write(*, '(a,f20.12)') 'Absolute error: ', abs_error
    write(*, '(a,f10.6,a)') 'Relative error: ', rel_error, ' %'
    
end program riemann_sin_area
