from Functions import *
from random import randint
import numpy as np

def Testy():
    # print(x2)
    # y = [26.0, 48.6, 12.6, 71.2, 74.8, 75.2,60.0]
    # x = [0,20,40,60,80,100,120]
    x = [1, 3, 5]
    y = [6, -2, 4]
    r, xx = FunkcjaSklejania(x, y)
    # xx = [i for i in np.arange(x[0], len(r)*0.25+x[0], 0.25)]
    # print(r)
    PlotData(x, y, "TEST", xx, r)
    N = 5


# x, y = LoadDataFromFile("./2018_paths/SpacerniakGdansk.csv")
d = 40
# x = [i for i in range(0,d, 5)]
# y = [randint(-20,20) for i in range(0, d, 5)]
x = [0, 5, 15]
y = [8, 18, 14]
N = 2

random_x, random_y = GetRandomXY(x, y, N)

yy = [ComputeLagrange(random_x, random_y, x_) for x_ in range(0, int(x[-1])+1)]
yy, xx = FunkcjaSklejania(x, y, N)
#xx = [i for i in range(0, int(x[-1])+1)]
PlotData(x, y, "Wykres wzniesień", xx, yy)

# Testy()

# x = [[0,0,1,2],
#      [0,1,2,3],
#      [1,2,3,4],
#      [0,0,0,1]]
#
# y = [3,2,1,4]
# a = [[1, 0, 0, 0, 0, 0, 0, 0], [1, 2, 4, 8, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 12], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 2, 4, 8], [0, 1, 4, 12, 0, -1, 0, 0], [0, 0, 2, 12, 0, 0, -2, 0]]
# b = [6, -2, 0, 0, -2, 4, 0, 0]
# print(linalg.solve(a, b))
