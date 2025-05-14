function conjugate_gradient_solver()
    % Define the system matrix A and right-hand side vector b
    A = [4, -1, 0; 
         -1, 4, -1; 
         0, -1, 4];
    b = [1; 2; 3];
    
    % Initial guess (can be zeros or any other vector)
    x0 = zeros(size(b));
    
    % Solve using our conjugate gradient implementation
    [x_cg, iter, residuals] = myConjugateGradient(A, b, x0);
    
    % Solve using MATLAB's built-in pcg for comparison
    [x_matlab, flag, relres, matlab_iter] = pcg(A, b, 1e-10, 100);
    
    % Display results
    fprintf('Our Conjugate Gradient Solution:\n');
    disp(x_cg');
    fprintf('Number of iterations: %d\n', iter);
    
    fprintf('\nMATLAB pcg Solution:\n');
    disp(x_matlab');
    fprintf('Number of iterations: %d\n', matlab_iter);
    
    fprintf('\nDirect Solution (A\\b):\n');
    x_direct = A\b;
    disp(x_direct');
    
    % Plot convergence
    figure;
    semilogy(residuals, 'LineWidth', 2);
    hold on;
    semilogy(relres*norm(b), 'r--', 'LineWidth', 2);
    title('Convergence of Conjugate Gradient Method');
    xlabel('Iteration');
    ylabel('Residual Norm (log scale)');
    legend('Our Implementation', 'MATLAB pcg');
    grid on;
    
    % Compute errors
    err_cg = norm(x_cg - x_direct);
    err_matlab = norm(x_matlab - x_direct);
    
    fprintf('\nError vs Direct Solution:\n');
    fprintf('Our CG Error: %e\n', err_cg);
    fprintf('MATLAB pcg Error: %e\n', err_matlab);
end

function [x, iter, residuals] = myConjugateGradient(A, b, x0)
    % Initialize variables
    x = x0;
    r = b - A*x;
    p = r;
    rsold = r'*r;
    
    max_iter = 100;
    tol = 1e-10;
    residuals = zeros(max_iter, 1);
    residuals(1) = norm(r);
    
    for iter = 1:max_iter
        Ap = A*p;
        alpha = rsold / (p'*Ap);
        x = x + alpha*p;
        r = r - alpha*Ap;
        rsnew = r'*r;
        residuals(iter+1) = norm(r);
        
        % Check for convergence
        if sqrt(rsnew) < tol
            residuals = residuals(1:iter+1);
            break;
        end
        
        p = r + (rsnew/rsold)*p;
        rsold = rsnew;
    end
end
