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

b = [0 for i in range(8)]
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
    A[3][1] = h2
    A[3][2] = pow(h2, 2)
    A[3][3] = pow(h2, 3)
    A[4][2] = 2*h1
    A[4][3] = 3*pow(h1, 2)
    A[5][3] = 6*h1
    A[7][7] = 6*h3

def FunkcjaSklejania(X_val, Y_val):
    # S0(x0) = f(x0)
    b[0] = Y_val[0]
    # S0(x1) = f(x1)
    b[1] = Y_val[1]
    # S1(x1) = f(x1)
    b[2] = Y_val[1]
    # S1(x2) = f(x2)
    b[3] = Y_val[2]

    setMatrixA(X_val[1]-X_val[0], X_val[2]-X_val[1], X_val[2]-X_val[1])
    r = linalg.solve(A, b)
    values = []
    for i in range(0, len(X_val)-1):
        for x in np.arange(int(X_val[i]), int(X_val[i+1]), 0.25):
            h = x-int(X_val[i])
            values.append(r[i*4+0] + r[i*4+1] * h + r[i*4+2] * pow(h, 2) + r[i*4+3] * pow(h, 3))
    h = int(X_val[-1])-int(X_val[i])
    values.append(r[i * 4 + 0] + r[i * 4 + 1] * h + r[i * 4 + 2] * pow(h, 2) + r[i * 4 + 3] * pow(h, 3))
    return values
