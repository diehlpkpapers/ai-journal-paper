

## Conjugate gradient

```bash
[ 93%] Building CXX object translation/cpp/CMakeFiles/cg_translated.dir/cg.cpp.o
/Users/diehlpk/git/ai-journal-paper/translation/cpp/cg.cpp:115:24: error: expected '}'
  115 |     cout << "Solution";
      |                        ^
/Users/diehlpk/git/ai-journal-paper/translation/cpp/cg.cpp:48:12: note: to match this '{'
   48 | int main() {
      |            ^
1 error generated.
```

## Heat equation solver 

### Parallel

```bash
[ 94%] Building CXX object translation/cpp/CMakeFiles/heat_shared_translated.dir/heat-shared.cpp.o
/Users/diehlpk/git/ai-journal-paper/translation/cpp/heat-shared.cpp:12:18: error: expected unqualified-id
   12 |     const double M_PI = 4 * atan(1);
      |                  ^
/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.1.sdk/usr/include/math.h:710:21: note: expanded from macro 'M_PI'
  710 | #define M_PI        3.14159265358979323846264338327950288   /* pi             */
      |                     ^
1 error generated.
make[2]: *** [translation/cpp/CMakeFiles/heat_shared_translated.dir/heat-shared.cpp.o] Error 1
make[1]: *** [translation/cpp/CMakeFiles/heat_shared_translated.dir/all] Error 2
make: *** [all] Error 2
```

