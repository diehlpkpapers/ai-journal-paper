% Define the heat equation function
% Computes the derivative of the solution u with respect to time t using the heat equation
function du_dt = heat_equation(u, dx, dt, alpha)
    % Get the number of grid points
    nx = length(u);
    % Initialize the derivative vector
    du_dt = zeros(nx, 1);
    
    % Compute the derivative using the heat equation formula
    for i = 2:nx-1
        du_dt(i) = alpha * (u(i+1) - 2*u(i) + u(i-1)) / dx^2;
    end
end

% Define the Euler step function
% Updates the solution u using the Euler method
function u_new = euler_step(u, du_dt, dt)
    % Update the solution using the Euler formula
    u_new = u + dt * du_dt;
end

% Define the parallel solver function
% Solves the 1D heat equation using the Euler method with parallel processing
function u = solve_heat_equation_parallel(u0, dx, dt, alpha, t_end, num_workers)
    % Get the number of grid points
    nx = length(u0);
    % Compute the number of time steps
    nt = ceil(t_end / dt);
    % Initialize the solution vector
    u = u0;
    
    % Create a parallel pool of workers
    parpool(num_workers);
    
    % Time-stepping loop
    for i = 1:nt
        % Compute the derivative using the heat equation
        du_dt = heat_equation(u, dx, dt, alpha);
        % Update the solution using the Euler method
        u = euler_step(u, du_dt, dt);
    end
    
    % Delete the parallel pool
    delete(gcp('nocreate'));
end

% Set up the problem parameters
L = 1.0;  % Length of the domain
nx = 100;  % Number of grid points
dx = L / (nx - 1);  % Spatial step size
t_end = 0.1;  % Final time
alpha =|

