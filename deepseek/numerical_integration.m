function sinx_riemann_area()
    % Main function to compute and validate the area under sin(x)
    
    % Integration bounds
    a = -pi;
    b = (2/3)*pi;
    
    % Number of rectangles (increase for better accuracy)
    n = 1000000;
    
    % Compute Riemann sum approximation
    approx_area = riemann_sum(a, b, n);
    
    % Compute exact integral value
    exact_area = exact_integral(a, b);
    
    % Calculate errors
    abs_error = abs(approx_area - exact_area);
    rel_error = (abs_error / abs(exact_area)) * 100;
    
    % Display results
    fprintf('Calculating area of sin(x) from %.6f to %.6f\n\n', a, b);
    fprintf('Riemann sum approximation (%,d rectangles): %.12f\n', n, approx_area);
    fprintf('Exact integral value: %.12f\n', exact_area);
    fprintf('Absolute error: %.12f\n', abs_error);
    fprintf('Relative error: %.6f %%\n\n', rel_error);
    
    % Optional visualization
    visualize(a, b);
end

function area = riemann_sum(a, b, n)
    % Compute Riemann sum using midpoint rule
    dx = (b - a)/n;          % Width of each rectangle
    x = a + dx/2:dx:b-dx/2;  % Midpoints of all intervals
    area = sum(sin(x)*dx);   % Sum of areas of all rectangles
end

function integral = exact_integral(a, b)
    % Compute exact integral of sin(x) from a to b
    integral = -cos(b) + cos(a);
end

function visualize(a, b)
    % Visualize the function and shaded area
    figure;
    x = linspace(a, b, 1000);
    y = sin(x);
    
    plot(x, y, 'b', 'LineWidth', 2);
    hold on;
    area(x, y, 'FaceColor', [0.7 0.8 1], 'EdgeColor', 'none');
    plot(x, zeros(size(x)), 'k--'); % x-axis
    title(sprintf('Area under sin(x) from %.2f to %.2f', a, b));
    xlabel('x');
    ylabel('sin(x)');
    grid on;
    hold off;
end
