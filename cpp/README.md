## Conjugate gradient solver

The clang compielr showed some errors messages which could be fixed bu sing the C++ 14 standard.

´```bash
[ 83%] Building CXX object cpp/CMakeFiles/cg_cpp.dir/cg.cpp.o
/Users/diehlpk/git/ai-journal-paper/cpp/cg.cpp:8:47: error: a space is required between consecutive right angle brackets (use '> >')
vector<double> matVecMult(vector<vector<double>> A, vector<double> x) {
                                              ^~
                                              > >
/Users/diehlpk/git/ai-journal-paper/cpp/cg.cpp:30:54: error: a space is required between consecutive right angle brackets (use '> >')
vector<double> conjugateGradient(vector<vector<double>> A, vector<double> b, double tol = 1e-5, int maxIter = 100) {
                                                     ^~
                                                     > >
/Users/diehlpk/git/ai-journal-paper/cpp/cg.cpp:61:25: error: a space is required between consecutive right angle brackets (use '> >')
    vector<vector<double>> A = {{4, -1, 0}, {-1, 4, -1}, {0, -1, 4}};
                        ^~
                        > >
/Users/diehlpk/git/ai-journal-paper/cpp/cg.cpp:61:28: error: non-aggregate type 'vector<vector<double> >' cannot be initialized with an initializer list
    vector<vector<double>> A = {{4, -1, 0}, {-1, 4, -1}, {0, -1, 4}};
                           ^   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/Users/diehlpk/git/ai-journal-paper/cpp/cg.cpp:62:20: error: non-aggregate type 'vector<double>' cannot be initialized with an initializer list
    vector<double> b = {1, 2, 3};
                   ^   ~~~~~~~~~
/Users/diehlpk/git/ai-journal-paper/cpp/cg.cpp:74:25: error: a space is required between consecutive right angle brackets (use '> >')
    vector<vector<double>> A_copy = A;
                        ^~
                        > >
6 errors generated.
make[2]: *** [cpp/CMakeFiles/cg_cpp.dir/cg.cpp.o] Error 1
make[1]: *** [cpp/CMakeFiles/cg_cpp.dir/all] Error 2
```

