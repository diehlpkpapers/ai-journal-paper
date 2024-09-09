## Conjungate gradient solver

```bash
[ 37%] Building Fortran object fortran/CMakeFiles/cg.dir/cg.f90.o
/Users/diehlpk/git/ai-journal-paper/fortran/cg.f90:31:6:

   31 |     Ap = MATMUL(A, p)
      |      1
Error: Symbol 'ap' at (1) has no IMPLICIT type; did you mean 'a'?
/Users/diehlpk/git/ai-journal-paper/fortran/cg.f90:23:3:

   23 |   p = r
      |   1
Error: Symbol 'p' at (1) has no IMPLICIT type
/Users/diehlpk/git/ai-journal-paper/fortran/cg.f90:20:3:

   20 |   r = b
      |   1
Error: Symbol 'r' at (1) has no IMPLICIT type
/Users/diehlpk/git/ai-journal-paper/fortran/cg.f90:43:11:

   43 |     rho_new = DOT_PRODUCT(r, r)
      |           1
Error: Symbol 'rho_new' at (1) has no IMPLICIT type
/Users/diehlpk/git/ai-journal-paper/fortran/cg.f90:92:8:

   92 |         A(j,i:) = A(j,i:) - factor * A(i,i:)
      |        1
Error: Dummy argument 'a' with INTENT(IN) in variable definition context (assignment) at (1)
/Users/diehlpk/git/ai-journal-paper/fortran/cg.f90:93:8:

   93 |         b(j) = b(j) - factor * b(i)
      |        1
Error: Dummy argument 'b' with INTENT(IN) in variable definition context (assignment) at (1)
make[2]: *** [fortran/CMakeFiles/cg.dir/cg.f90.o] Error 1
make[1]: *** [fortran/CMakeFiles/cg.dir/all] Error 2
make: *** [all] Error 2
```

