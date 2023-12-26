### Score
15.00 / 16.00

### Fixes
2/23 15:42 problem 0b.
In assign.pdf, ~ that we flip it 8 times and ... → ~ that we flip it 5 times and ...

2/23 21:59 problem 1.
In grader.py, there was an error in generating v1 and v2 of hidden cases related to dense vector because the "randvec()" function overlapped. Thus, the randvec() functions was renamed to "randDenseVec()" and "randSparseVec()".