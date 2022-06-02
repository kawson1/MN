from copy import deepcopy
from random import sample
from numpy.linalg import linalg
import numpy as np

import matplotlib.pyplot as plt


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


# n - num of intervals
def FunkcjaSklejania(X_val, Y_val, n=2):
    if len(X_val) <= n:
        print("Za mało wartości X, za dużo przedziałów!")
        return 0
    # GENERATE X EDGES
    x_edges_idx = [i for i in sample(range(1, len(X_val)-1), n-1)]
    x_edges_idx.insert(0, 0)
    x_edges_idx.append(len(X_val)-1)
    x_edges_idx.sort()
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
    A_[2][2] = 1
    A_[3][4*(n-1)+2] = 2
    hn = X_val[x_edges_idx[-1]] - X_val[x_edges_idx[-2]]
    A_[3][4*(n-1)+3] = 6*hn

    for j in range(1, n):
        # S0: xn - x(n-1)
        h = X_val[x_edges_idx[j+1]] - X_val[x_edges_idx[j]]
        # an
        A_[j * 4 + 0][j * 4 + 0] = 1
        # an*bn(h)...
        A_[j * 4 + 1][j * 4 + 0] = 1
        A_[j * 4 + 1][j * 4 + 1] = h
        A_[j * 4 + 1][j * 4 + 2] = pow(h, 2)
        A_[j * 4 + 1][j * 4 + 3] = pow(h, 3)
        # b(n-1)+2c(n-1)*h...-bn
        h = X_val[x_edges_idx[j]] - X_val[x_edges_idx[j-1]]
        A_[j * 4 + 2][(j - 1) * 4 + 1] = 1
        A_[j * 4 + 2][(j - 1) * 4 + 2] = 2 * h
        A_[j * 4 + 2][(j - 1) * 4 + 3] = 3 * pow(h, 2)
        A_[j * 4 + 2][j * 4 + 1] = -1
        # 2*c(n-1)*h-2cn
        A_[j * 4 + 3][(j - 1) * 4 + 2] = 2
        A_[j * 4 + 3][(j - 1) * 4 + 3] = 6 * h
        A_[j * 4 + 3][j * 4 + 2] = -2

        # S(j-1)(xS) = f(xS)
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
    x_values = []
    for j in range(0, n):
        for x in range(int(X_val[x_edges_idx[j]]), int(X_val[x_edges_idx[j+1]])):
            h = x-int(X_val[x_edges_idx[j]])
            values.append(r[j*4+0] + r[j*4+1] * h + r[j*4+2] * pow(h, 2) + r[j*4+3] * pow(h, 3))
            x_values.append(x)
    values.append(Y_val[-1])
    x_values.append(int(X_val[-1]))
    return x_values, values


def PlotMetodaSklejania(x, y, N):
    xx, yy = FunkcjaSklejania(x, y, N)
    PlotData(x, y, "Wykres wzniesień - metoda sklejania", xx, yy)
    return


def PlotMetodaLagrange(x, y, N):
    random_x, random_y = GetRandomXY(x, y, N)
    yy = [ComputeLagrange(random_x, random_y, x_) for x_ in range(0, int(x[-1])+1)]
    xx = [i for i in range(0, int(x[-1])+1)]
    PlotData(x, y, "Wykres wzniesień - metoda Lagrange'a", xx, yy)
    return


def GetColumn(A, col):
    return [row[col] for row in A]


def Pivot(A, b):
    swap_list = []
    for row in range(len(A)):
        for col in range(len(A)):
            if A[row][row] == 0:
                max_idx = 0
                max = A[0][col]
                for i in range(len(A)):
                    if max < A[i][col]:
                        max_idx = i
                        max = A[i][col]
                A[row][col], A[max_idx][col] = A[max_idx][col]



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
