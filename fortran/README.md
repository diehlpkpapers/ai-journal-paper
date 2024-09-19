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

## Parallel heat eqution

### Parallel using OpenMP

````bash
[ 41%] Building Fortran object fortran/CMakeFiles/heat_shared.dir/heat-shared.f90.o
/vast/home/diehlpk/ai-journal-paper/fortran/heat-shared.f90:16:6:

   DO i = 1, NX
      1
Error: Symbol ‘i’ at (1) has no IMPLICIT type
/vast/home/diehlpk/ai-journal-paper/fortran/heat-shared.f90:21:6:

   DO j = 1, NT
      1
Error: Symbol ‘j’ at (1) has no IMPLICIT type
/vast/home/diehlpk/ai-journal-paper/fortran/heat-shared.f90:17:19:

     u(i) = SIN(M_PI * (i - 1) * DX)  ! Initial condition
                   1
Error: Symbol ‘m_pi’ at (1) has no IMPLICIT type
make[2]: *** [fortran/CMakeFiles/heat_shared.dir/build.make:75: fortran/CMakeFiles/heat_shared.dir/heat-shared.f90.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:176: fortran/CMakeFiles/heat_shared.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
```
