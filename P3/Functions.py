from random import sample
from numpy.linalg import linalg
import numpy as np

import matplotlib.pyplot as plt

A = [[1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0],
     [1, 2, 4, 8, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 2, 4, 8],
     [0, 1, 4, 12, 0, -1, 0, 0],
     [0, 0, 2, 12, 0, 0, -2, 0],
     [0, 0, 2, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 2, 12]]

# b = [0 for i in range(8)]
def LoadDataFromFile(filename):
    x = []
    y = []
    with open(filename) as f:
        for value in f.readlines():
            value = value.replace("\n", "").split(',')
            x.append(float(value[0]))
            y.append(float(value[1]))
    return x, y


# Args - X and Y values
def PlotData(x ,y, title, *args):
    plt.plot(x, y, '.')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    for i in range(len(args)//2):
        plt.plot(args[i], args[i+1])
    plt.show()


# n - HOW MUCH X VALUES TO TAKE
def GetRandomXY(x_values, y_values, n):
    randomX = [x_values[0]]
    randomY = [y_values[0]]
    if n > 2 and n <= len(x_values):
        for idx in sample(range(1, len(x_values)-1), n-2):
            randomX.append(x_values[idx])
            randomY.append(y_values[idx])
    randomX.append(x_values[-1])
    randomY.append(y_values[-1])
    return randomX, randomY


def ComputeLagrange(x_values, y_values, x):
    result = 0
    for i in range(len(x_values)):
        phi_i = 1
        for j in range(len(x_values)):
            if i != j:
                phi_i *= (x-x_values[j])/(x_values[i]-x_values[j])
        result += phi_i * y_values[i]
    return result


# n - NUMBER OF RANGES (MAX N = LEN(X)-1)
# def FunkcjaSklejania(x_values, y_values, n):
#     x_range = len(x_values) // n
#     # CURRENT RANGE
#     curr_S = 0
#     curr_x = 0
#     S = []
#     for i in range(0, n-1):
#         b[0] = y_values[curr_S]
#         b[1] = b[2] = y_values[curr_S + x_range]
#         b[3] = y_values[curr_S + 2*x_range]
#         # r = [a0, b0, c0, d0, a1, b1, c1, d1]
#         r = linalg.solve(A, b)
#         print()
#         for x in range(curr_S, curr_S+x_range):
#             bracket = x+curr_x-x_values[i]
#             S0 = r[0] + r[1]*bracket + r[2]*pow(bracket, 2) + r[3]*pow(bracket, 3)
#             S.append(S0)
#             curr_x += 1
#         # -1 BECAUSE IF eg. X_RANGE=5 -> CURR_X=0 -> loop for 0...!19!
#         curr_S += x_range
#     for x in range(curr_x, curr_x + x_range):
#         bracket = x+curr_x - x_values[i]
#         S0 = r[4] + r[5] * bracket + r[6] * pow(bracket, 2) + r[7] * pow(bracket, 3)
#         S.append(S0)
#     return S

# h1    -   S0(x1-x0)
# h2    -   S1(x2-x1)
# h3    -   Sn-1(xn-x(n-1))
def setMatrixA(h1, h2, h3):
    A[2][1] = h1
    A[2][2] = pow(h1, 2)
    A[2][3] = pow(h1, 3)
    A[3][5] = h2
    A[3][6] = pow(h2, 2)
    A[3][7] = pow(h2, 3)
    A[4][2] = 2*h1
    A[4][3] = 3*pow(h1, 2)
    A[5][3] = 6*h1
    A[7][7] = 6*h3

# def setMatrix(matrix):
#     for i in range(len(matrix)/4):
#         matrix[i*4+0][i*4+0] = 1
#
#         matrix[i*4+1][i*4+0] = 1
#         # *h
#         matrix[i*4+1][i*4+1] = 1
#         # *h^2
#         matrix[i*4+1][i*4+2] = 1
#         # *h^3
#         matrix[i*4+1][i*4+3] = 1
#
#         matrix[i*4+2][i*4+4] = 1

# n - num of intervals
def FunkcjaSklejania(X_val, Y_val, n=2):
    if len(X_val) <= n:
        print("Za mało wartości X, za dużo przedziałów!")
        return 0
    # GENERATE X EDGES
    x_edges_idx = [i for i in sample(range(1, len(X_val)-1), n-1)]
    x_edges_idx.insert(0, 0)
    x_edges_idx.append(len(X_val)-1)
    # EACH INTERVAL TAKE 4 EQUATIONS
    A_ = [[0 for i in range(n*4)] for i in range(n*4)]
    b = []

    # INIT B
    # S0(x0) = f(x0)
    b.append(Y_val[0])
    # S0(x1) = f(x1)
    b.append(Y_val[x_edges_idx[1]])
    # EDGES S''(x0) = 0 && S''(n-1)(xn) = 0
    b.append(0)
    b.append(0)

    h0 = X_val[x_edges_idx[1]] - X_val[x_edges_idx[0]]
    # INIT A
    A_[0][0] = 1
    A_[1][0] = 1
    A_[1][1] = h0
    A_[1][2] = pow(h0, 2)
    A_[1][3] = pow(h0, 3)
    A_[2][2] = 2
    A_[3][4*(n-1)+2] = 2
    hn = X_val[x_edges_idx[-1]] - X_val[x_edges_idx[-2]]
    A_[3][4*(n-1)+3] = 6*hn

    for j in range(1, n):
        # S0: xn - x(n-1)
        h = X_val[x_edges_idx[j]] - X_val[x_edges_idx[j - 1]]
        # an
        A_[j * 4 + 0][j * 4 + 0] = 1
        # an*bn(h)...
        A_[j * 4 + 1][j * 4 + 0] = 1
        A_[j * 4 + 1][j * 4 + 1] = h
        A_[j * 4 + 1][j * 4 + 2] = pow(h, 2)
        A_[j * 4 + 1][j * 4 + 3] = pow(h, 3)
        # b(n-1)+2c(n-1)*h...-bn
        A_[j * 4 + 2][(j - 1) * 4 + 1] = 1
        A_[j * 4 + 2][(j - 1) * 4 + 2] = 2 * h
        A_[j * 4 + 2][(j - 1) * 4 + 3] = 3 * pow(h, 2)
        A_[j * 4 + 2][j * 4 + 1] = -1
        # 2*c(n-1)*h-2cn
        A_[j * 4 + 3][(j - 1) * 4 + 2] = 2
        A_[j * 4 + 3][(j - 1) * 4 + 3] = 6 * h
        A_[j * 4 + 3][j * 4 + 2] = -2

        # Si(xS) = f(xS)
        b.append(Y_val[x_edges_idx[j]])
        # Si(xS+1) = f(xS+1)
        b.append(Y_val[x_edges_idx[j+1]])
        # Si-1(xS) - Si(xS) = 0
        b.append(0)
        # Si-1(xS) - Si(xS) = 0
        b.append(0)

        # setMatrixA(X_val[1]-X_val[0], X_val[2]-X_val[1], X_val[2]-X_val[1])
    r = linalg.solve(A_, b)
    values = []
    for j in range(0, n):
        for x in np.arange(int(X_val[x_edges_idx[j]]), int(X_val[x_edges_idx[j+1]]), 0.25):
            h = x-int(X_val[x_edges_idx[j]])
            values.append(r[j*4+0] + r[j*4+1] * h + r[j*4+2] * pow(h, 2) + r[j*4+3] * pow(h, 3))
    values.append(Y_val[-1])
    return values
