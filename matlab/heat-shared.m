% Define the heat equation function
function du_dt = heat_equation(u, dx, dt, alpha)
    nx = length(u);
    du_dt = zeros(nx, 1);
    
    for i = 2:nx-1
        du_dt(i) = alpha * (u(i+1) - 2*u(i) + u(i-1)) / dx^2;
    end
end

% Define the Euler step function
function u_new = euler_step(u, du_dt, dt)
    u_new = u + dt * du_dt;
end

% Define the parallel solver function
function u = solve_heat_equation_parallel(u0, dx, dt, alpha, t_end, num_workers)
    nx = length(u0);
    nt = ceil(t_end / dt);
    u = u0;
    
    parpool(num_workers);
    
    for i = 1:nt
        du_dt = heat_equation(u, dx, dt, alpha);
        u = euler_step(u, du_dt, dt);
    end
    
    delete(gcp('nocreate'));
end

% Set up the problem parameters
L = 1.0;
nx = 100;
dx = L / (nx - 1);
t_end = 0.1;
alpha = 0.1;
num_workers = 4;

% Set up the initial condition
x = 0:dx:L;
u0 = sin(pi * x);

% Solve the heat equation in parallel
u = solve_heat_equation_parallel(u0, dx, dt, alpha, t_end, num_workers);

% Plot the solution
plot(x, u);
xlabel('x');
ylabel('u(x,t)');
title('Solution of the 1D Heat Equation');

% Validate the solution
exact_solution = @(x, t, alpha) sin(pi * x) * exp(-alpha * pi^2 * t);

t = t_end;
u_exact = exact_solution(x, t, alpha);

hold on;
plot(x, u_exact, 'r');
legend('Numerical Solution', 'Exact Solution');
