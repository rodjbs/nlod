# Normalised Levenshtein Distance for OD matrices (NLOD)

# Behara K, Bhaskar A, Chung E
# "A novel approach for the structural comparison
#  of origin-destination matrices: Levenshtein distance"
# https://doi.org/10.1016/j.trc.2020.01.005

# Implementation by Rodrigo Simões

import numpy as np

def nlod(matrix1, matrix2):
    """Calculate NLOD between two OD matrices"""

    assert matrix1.ndim == 1 or matrix1.ndim == 2, "invalid matrix dimension"
    assert matrix2.ndim == 1 or matrix2.ndim == 2, "invalid matrix dimension"
    assert matrix1.size == matrix2.size, "matrices must have the same size"
    matrix_size = matrix1.size ** (1/2)
    assert matrix_size == int(matrix_size), "matrices must be square"
    
    if matrix1.ndim == 2:
        matrix1 = matrix1.flatten()
    if matrix2.ndim == 2:
        matrix2 = matrix2.flatten()

    return _nlod(matrix1, matrix2)

def _nlod(matrix1, matrix2):
    matrix_size = int(matrix1.size ** (1/2))
    res = 0
    for n in range(matrix_size):
        line1 = matrix1[matrix_size*n:matrix_size*(n+1)]
        line2 = matrix2[matrix_size*n:matrix_size*(n+1)]
        res += _local_nlod(line1, line2)
    res = res / matrix_size
    return res

def _local_nlod(line1, line2):
    X = _order_line(line1)
    Y = _order_line(line2)

    M = line1.size
    L = np.ndarray((M+1,M+1))

    # fill first row and first column
    L[0,0] = 0
    for i in range(1, M+1):
        L[0,i] = L[0,i-1] + X[i-1][1]
        L[i,0] = L[i-1,0] + Y[i-1][1]

    # fill remaining cells
    for i in range(1, M+1):
        for j in range(1, M+1):
            DX, AX = X[j-1]
            DY, AY = Y[i-1]
            
            C = abs(AX - AY) if DX == DY else abs(AX + AY)

            L[i,j] = min(
                L[i-1,j] + AY,
                L[i,j-1] + AX,
                L[i-1,j-1] + C
            )

    # result
    local_lod = L[M,M]
    local_nlod = local_lod / sum([e[1] for e in X + Y])
    return local_nlod

def _order_line(line):
    return sorted(enumerate(line), key=lambda e: e[1], reverse=True)

def _test_nlod():
    X = np.array([
        3,4,6,10,
        7,4,5,11,
        12,8,5,6,
        13,7,9,6
    ])

    Y = np.array([
        10,9,12,16,
        17,10,13,11,
        11,14,12,18,
        12,13,19,15
    ])

    print(nlod(X,Y))

if __name__ == "__main__":
    _test_nlod()
