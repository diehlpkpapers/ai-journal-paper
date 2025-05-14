# Errors in numerical_integration.r


Error in sprintf("Riemann sum approximation (%,d rectangles): %.12f\n", : invalid format '%,d'; use format %f, %e, %g or %a for numeric objects
Traceback:

1. sprintf("Riemann sum approximation (%,d rectangles): %.12f\n", 
 .     n, approx_area)
2. .handleSimpleError(function (cnd) 
 . {
 .     watcher$capture_plot_and_output()
 .     cnd <- sanitize_call(cnd)
 .     watcher$push(cnd)
 .     switch(on_error, continue = invokeRestart("eval_continue"), 
 .         stop = invokeRestart("eval_stop"), error = NULL)
 . }, "invalid format '%,d'; use format %f, %e, %g or %a for numeric objects", 
 .     base::quote(sprintf("Riemann sum approximation (%,d rectangles): %.12f\n", 
 .         n, approx_area)))


# Errors in phe.r
  Error in library(doParallel): there is no package called ‘doParallel’
  Traceback:
  
  1. stop(packageNotFoundError(package, lib.loc, sys.call()))

# Errors in phe.m
 Error: File: parallel_heat_equation.m Line: 33 Column: 9
 spmd statements cannot be used inside parfor-loops. For more
 information, see Parallel for Loops in MATLAB, "Nested spmd
 Statements".
 
