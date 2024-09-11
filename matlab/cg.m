% Define the linear equation system
A = [4 -1 0; -1 4 -1; 0 -1 4];  % matrix
b = [1; 2; 3];  % right-hand side vector

% Define the tolerance and maximum number of iterations
tol = 1e-6;
max_iter = 100;

% Initialize the solution vector
x = zeros(size(b));

% Conjugate Gradient solver
r = b - A*x;  % initial residual
p = r;  % initial search direction
rho = dot(r, r);  % initial residual norm
for k = 1:max_iter
    Ap = A*p;  % matrix-vector product
    alpha = rho / dot(p, Ap);  % step size
    x = x + alpha*p;  % update solution
    r = r - alpha*Ap;  % update residual
    rho_new = dot(r, r);  % new residual norm
    if sqrt(rho_new) < tol
        break;
    end
    beta = rho_new / rho;  % update beta
    p = r + beta*p;  % update search direction
    rho = rho_new;  % update residual norm
end

% Validate the solution
residual = norm(b - A*x);
fprintf('Residual norm: %e\n', residual);
if residual < tol
    fprintf('Solution validated!\n');
else
    fprintf('Solution not validated!\n');
end

% Compare with built-in solver (optional)
x_builtin = A\b;
fprintf('Built-in solver solution: \n');
disp(x_builtin);
fprintf('CG solver solution: \n');
disp(x);

