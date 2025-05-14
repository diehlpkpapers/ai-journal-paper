function parallel_heat_equation()
    % Parameters
    alpha = 0.1;       % Thermal diffusivity
    L = 1;             % Length of the domain
    T = 0.1;           % Total simulation time
    n = 100;           % Number of spatial points
    nt = 1000;         % Number of time steps
    
    % Derived parameters
    h = L/(n-1);       % Spatial step size
    dt = T/nt;         % Time step size
    x = linspace(0, L, n)';  % Spatial grid
    
    % Initial condition
    u0 = sin(pi*x);
    
    % Analytical solution
    u_analytical = @(t) sin(pi*x) * exp(-alpha*pi^2*t);
    
    % Parallel setup
    pool = gcp();  % Start parallel pool if not already running
    num_workers = pool.NumWorkers;
    
    % Divide work among workers
    chunk_size = ceil(n/num_workers);
    
    % Solve heat equation in parallel
    u = u0;
    tic;
    parfor (k = 1:nt, num_workers)
        u_new = u;
        % Each worker handles its chunk of the spatial domain
        spmd
            worker_range = (labindex-1)*chunk_size + 1 : min(labindex*chunk_size, n);
            for i = worker_range
                if i == 1
                    % Left boundary (Dirichlet)
                    u_new(i) = 0;
                elseif i == n
                    % Right boundary (Dirichlet)
                    u_new(i) = 0;
                else
                    % Interior points (central difference)
                    u_new(i) = u(i) + alpha*dt/h^2 * (u(i+1) - 2*u(i) + u(i-1));
                end
            end
        end
        u = u_new;
    end
    compute_time = toc;
    
    % Gather results from all workers
    u_final = u;
    
    % Validation
    u_exact = u_analytical(T);
    error = norm(u_final - u_exact, 'inf');
    
    % Display results
    fprintf('Parallel 1D Heat Equation Solver\n');
    fprintf('Spatial points: %d, Time steps: %d\n', n, nt);
    fprintf('Computation time: %.4f seconds\n', compute_time);
    fprintf('Maximum error at t=%.2f: %.6e\n', T, error);
    
    % Plot results
    figure;
    plot(x, u_final, 'b-', 'LineWidth', 2, 'DisplayName', 'Numerical');
    hold on;
    plot(x, u_exact, 'r--', 'LineWidth', 2, 'DisplayName', 'Analytical');
    xlabel('Position x');
    ylabel('Temperature u(x)');
    title(sprintf('1D Heat Equation at t=%.2f (α=%.1f)', T, alpha));
    legend('Location', 'best');
    grid on;
    
    % Plot error distribution
    figure;
    plot(x, abs(u_final - u_exact), 'LineWidth', 2);
    xlabel('Position x');
    ylabel('Absolute Error');
    title('Error Distribution');
    grid on;
end
