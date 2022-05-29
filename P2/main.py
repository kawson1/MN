import math
import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt


def getDiagonal(matrix, level=0):
    A = matrix
    diag = [[0 for i in range(0, len(A[0]))] for i in range(0, len(A))]
    for i in range(0, len(A) - abs(level)):
        if level == 0:
            diag[i][i] = A[i][i]
        elif level < 0:
            diag[i + abs(level)][i] = A[i + abs(level)][i]
        else:
            diag[i][i + level] = A[i][i + level]
    return diag


def getLowerTriangle(matrix):
    A = matrix
    N = len(matrix)
    lower = [[0 for i in range(0, N)] for i in range(0, N)]
    for row in range(1, N):
        for col in range(0, row):
            lower[row][col] = A[row][col]
    return lower


def getLowerTriangleWithDiagonal(matrix):
    A = matrix
    N = len(matrix)
    lower = [[0 for i in range(0, N)] for i in range(0, N)]
    for row in range(1, N):
        for col in range(0, row):
            lower[row][col] = A[row][col]
    addMatrixes(lower, getDiagonal(matrix))
    return lower


def getUpperTriangle(matrix):
    A = matrix
    N = len(matrix)
    upper = [[0 for i in range(0, N)] for i in range(0, N)]
    for col in range(1, N):
        for row in range(0, col):
            upper[row][col] = A[row][col]
    return upper


def getUpperTriangleWithDiagonal(matrix):
    A = matrix
    N = len(matrix)
    upper = [[0 for i in range(0, N)] for i in range(0, N)]
    for col in range(1, N):
        for row in range(0, col):
            upper[row][col] = A[row][col]
    return addMatrixes(upper, getDiagonal(matrix))


def transposedMatrix(A):
    result = [[0 for i in range(0, len(A))] for i in range(0, len(A))]
    for row in range(0, len(A)):
        for col in range(0, len(A)):
            result[col][row] = A[row][col]
    return result

# def inverseMatrix(A):
#     L, U = LU(A)


def multiplicationMatrix(A, B):
    if isinstance(B[0], list):
        columns = len(B[0])
        rows = len(B)
        result = [[0 for i in range(0, columns)] for i in range(0, len(A))]
    else:
        columns = len(B)
        rows = 1
        result = [0 for i in range(0, len(A))]
    if len(A[0]) != columns:
        print("ZLE ROZMIARY MACIERZY!")
        return
    for row in range(0, len(A)):
        for col in range(0, rows):
            for i in range(0, columns):
                if rows == 1:
                    result[row] += A[row][i] * B[i]
                else:
                    result[row][col] += A[row][i] * B[i][col]
    return result


def multiplicationMatrixByValue(A, value):
    for row in range(0, len(A)):
        for col in range(0, len(A)):
            A[row][col] *= value
    return A


def addMatrixes(A, B):
    if isinstance(A[0], list) and isinstance(A[0], list):
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            print("ZLE ROZMIARY MACIERZY!")
            return
        for row in range(0, len(A)):
            for col in range(0, len(A)):
                A[row][col] += B[row][col]
        return A
    # 1D matrix
    else:
        for i in range(0, len(A)):
            A[i] += B[i]
            return A

def subMatrixes(A, B):
    if isinstance(A[0], list) and isinstance(A[0], list):
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            print("ZLE ROZMIARY MACIERZY!")
            return
        for row in range(0, len(A)):
            for col in range(0, len(A)):
                A[row][col] -= B[row][col]
        return A
    # 1D matrix
    else:
        for i in range(0, len(A)):
            A[i] -= B[i]
        return A


def minValue(array):
    min = array[0]
    for i in range(0, len(array)):
        if array[i] < min:
            min = array[i]
    return min


def getLUValues(A):
    m = len(A)
    U = deepcopy(A)
    L = [[0 for i in range(0, len(A))] for i in range(0, len(A))]
    for i in range(0, len(A)):
        L[i][i] = 1
    for col in range(0, m-1):
        for row in range(col+1, m):
            L[row][col] = U[row][col]/U[col][col]
            for k in range(col, m):
                U[row][k] = U[row][k] - L[row][col] * U[col][k]
    return L, U


def LU(A, b):
    # UxY = b -> Y
    # LxX = Y -> X
    L, U = getLUValues(A)
    Y = [0 for i in range(0, len(L))]
    X = [0 for i in range(0, len(L))]
    # compute Y
    Y[0] = b[0] / L[0][0]
    for i in range(1, len(L)):
        suma = 0
        for j in range(0, i):
            suma += L[i][j] * Y[j]
        Y[i] = (b[i] - suma) / L[i][i]
    # compute result X
    X[len(L)-1] = Y[len(L)-1] / U[len(L)-1][len(L)-1]
    for i in range(len(L)-2, -1, -1):
        suma = 0
        for j in range(i+1, len(L)):
            suma += U[i][j] * X[j]
        X[i] = (Y[i] - suma) / U[i][i]
    return X


def inverseDiagonal(D):
    for i in range(0, len(D)):
        if D[i][i] == 0:
            continue
        D[i][i] = 1 / D[i][i]
    return D


def norm(M, b, r):
    Mr = multiplicationMatrix(M, r)
    # print("M : %s "% M)
    # print("R : %s" % r)
    # print("Mr : %s "% Mr)
    # print("b : %s "% b)
    return subMatrixes(Mr, b)


def jacobyMethod(A, b):
    X = [1 for i in range(len(b))]
    iterations = 0
    while True:
        old_X = [X[i] for i in range(len(b))]
        iterations+=1
        for row in range(len(A)):
            suma = 0
            for i in range(len(A)):
                if row != i:
                    suma += A[row][i]*old_X[i]
            X[row] = (b[row] - suma) / A[row][row]
        res = [b[i] for i in range(len(b))]
        for row in range(len(b)):
            for i in range(len(b)):
                res[row] -= A[row][i] * X[i]
        if (max(res) < pow(10, -9) and abs(min(res)) < pow(10, -9)) or iterations > 10000:
            if iterations > 10000:
                print("Metoda nie zbiega się.")
            return X, iterations


def gaussMethod(A, b):
    X = [1 for i in range(len(b))]
    iterations = 0
    while True:
        iterations+=1
        for row in range(len(A)):
            suma = 0
            for i in range(len(A)):
                if row != i:
                    suma += A[row][i]*X[i]
            X[row] = (b[row] - suma) / A[row][row]
        res = [b[i] for i in range(len(b))]
        for row in range(len(b)):
            for i in range(len(b)):
                res[row] -= A[row][i] * X[i]
        if (max(res) < pow(10, -9) and abs(min(res)) < pow(10, -9)) or iterations > 10000:
            if iterations > 10000:
                print("Metoda nie zbiega się.")
            return X, iterations


def inverseMatrix(A):
    L, U = getLUValues(A)
    I = [[0 for i in range(0, len(A))] for i in range(0, len(A))]
    X = [[0 for i in range(0, len(A))] for i in range(0, len(A))]

    for i in range(0, len(A)):
        I[i][i] = 1
    for k in range(0, len(A)):
        X_col = LU(L, U, I[k])
        for i in range(0, len(A)):
            X[i][k] = X_col[i]
    return X

def ZadanieAB():
    N = 948
    a1 = 9
    a2 = a3 = -1
    # A [ROW] [COL]
    A = [[0 for ROW in range(0, N)] for COL in range(0, N)]
    for i in range(0, N):
        A[i][i] = a1
        if i < N - 1:
            A[i + 1][i] = a2
            A[i][i + 1] = a2
        if i < N - 2:
            A[i + 2][i] = a3
            A[i][i + 2] = a3

    b = [0 for i in range(0, N)]
    for n in range(0, N):
        b[n] = math.sin(n * 5)


    time_j = time.time()
    X_j, iterations_j = jacobyMethod(A, b)
    time_j = time.time() - time_j
    time_g = time.time()
    X_g, iterations_g = gaussMethod(A, b)
    time_g = time.time() - time_g
    time_l = time.time()
    X_l = LU(A, b)
    time_l = time.time() - time_l

    print("~~~~~~~~~~~~   Jacobi method   ~~~~~~~~~~~~")
    print(f"Iterations: {iterations_j}")
    print(f"X values  : {X_j}")
    print(f"Res. norm : {norm(A, b, X_j)}")
    print(f"Time  : {time_j}")
    print()
    print("~~~~~~~~~~~~   Gauss method   ~~~~~~~~~~~~")
    print(f"Iterations: {iterations_g}")
    print(f"X values  : {X_g}")
    print(f"Res. norm : {norm(A, b, X_g)}")
    print(f"Time  : {time_g}")
    print("~~~~~~~~~~~~   LU method   ~~~~~~~~~~~~")
    print(f"X values  : {X_l}")
    print(f"Res. norm : {norm(A, b, X_l)}")
    print(f"Time  : {time_l}")


def ZadanieCD():
    N = 948
    A = [[0 for ROW in range(0, N)] for COL in range(0, N)]
    a1 = 3
    a2 = a3 = -1
    for i in range(0, N):
        A[i][i] = a1
        if i < N - 1:
            A[i + 1][i] = a2
            A[i][i + 1] = a2
        if i < N - 2:
            A[i + 2][i] = a3
            A[i][i + 2] = a3
    b = [0 for i in range(0, N)]
    for n in range(0, N):
        b[n] = math.sin(n * 5)

    time_lu = time.time()
    X_LU = LU(A, b)
    time_lu = time.time() - time_lu
    print("~~~~~~~~~~~~   LU method   ~~~~~~~~~~~~")
    print(f"X values  : {X_LU}")
    print(f"Res. norm : {norm(A, b, X_LU)}")
    print(f"Time  : {time_lu}")

    # time_j = time.time()
    # X_j, iterations_j = jacobyMethod(A, b)
    # time_j = time.time() - time_j
    # time_g = time.time()
    # X_g, iterations_g = gaussMethod(A, b)
    # time_g = time.time() - time_g

    # print("~~~~~~~~~~~~   Jacobi method   ~~~~~~~~~~~~")
    # print(f"Iterations: {iterations_j}")
    # print(f"X values  : {X_j}")
    # print(f"Time  : {time_j}")
    # print()
    # print("~~~~~~~~~~~~   Gauss method   ~~~~~~~~~~~~")
    # print(f"Iterations: {iterations_g}")
    # print(f"X values  : {X_g}")
    # print(f"Time  : {time_g}")


def ZadanieE():
    a1 = 9
    a2 = a3 = -1
    N_arr = [100, 500, 1000, 3000]
    jacobi_times = [0 for i in range(len(N_arr))]
    gauss_times = [0 for i in range(len(N_arr))]
    LU_times = [0 for i in range(len(N_arr))]
    for n_idx in range(len(N_arr)):
        N = N_arr[n_idx]
        # A [ROW] [COL]
        A = [[0 for ROW in range(0, N)] for COL in range(0, N)]
        for i in range(0, N):
            A[i][i] = a1
            if i < N - 1:
                A[i + 1][i] = a2
                A[i][i + 1] = a2
            if i < N - 2:
                A[i + 2][i] = a3
                A[i][i + 2] = a3
        b = [0 for i in range(0, N)]
        for n in range(0, N):
            b[n] = math.sin(n * 5)

        time_j = time.time()
        X_j, iterations_j = jacobyMethod(A, b)
        time_j = time.time() - time_j
        time_g = time.time()
        X_g, iterations_g = gaussMethod(A, b)
        time_g = time.time() - time_g
        time_lu = time.time()
        X_lu = LU(A, b)
        time_lu = time.time() - time_lu
        jacobi_times[n_idx] = time_j
        gauss_times[n_idx] = time_g
        LU_times[n_idx] = time_lu
    fig, ax = plt.subplots()
    jacobi_line = ax.plot(N_arr, jacobi_times, label="Jacobi")
    gauss_line = ax.plot(N_arr, gauss_times, label="Gauss")
    LU_line = ax.plot(N_arr, LU_times, label="LU")
    plt.title("Method execution time")
    ax.legend()
    plt.show()


ZadanieAB()
# ZadanieCD()
# ZadanieE()