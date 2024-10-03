
## Conjugate gradient

```bash
python3 cg.py
Traceback (most recent call last):
  File "/Users/diehlpk/git/ai-journal-paper/translation/python/cg.py", line 37, in <module>
    r -= alpha * Ap
numpy._core._exceptions.UFuncTypeError: Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```
```bash
 python3 cg-fixed.py
Solution using Conjugate Gradient solver:  [0.46428571 0.85714286 0.96428571]
Traceback (most recent call last):
  File "/Users/diehlpk/git/ai-journal-paper/translation/python/cg-fixed.py", line 47, in <module>
    x_ge = gaussian_elimination(A.copy(), b.copy())
  File "/Users/diehlpk/git/ai-journal-paper/translation/python/cg-fixed.py", line 14, in gaussian_elimination
    A[j,:] -= factor * A[i, :]
numpy._core._exceptions.UFuncTypeError: Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```
