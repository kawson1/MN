from Functions import *
from random import randint
import numpy as np

def Testy():
    # print(x2)
    # y = [26.0, 48.6, 12.6, 71.2, 74.8, 75.2,60.0]
    # x = [0,20,40,60,80,100,120]
    x = [i for i in range(0, 9, 4)]
    y = [6,-2, 4]
    print(x)
    r = FunkcjaSklejania(x, y)
    xx = [i for i in np.arange(0, len(r)*0.25, 0.25)]
    # print(r)
    PlotData(x, y, "TEST", xx, r)
    N = 5
    # sadas






# x, y = LoadDataFromFile("./2018_paths/SpacerniakGdansk.csv")
# N = 14
#
# random_x, random_y = GetRandomXY(x, y, N)
#
# yy = [ComputeLagrange(random_x, random_y, x_) for x_ in range(0, int(x[-1])+1)]
# xx = [i for i in range(0, int(x[-1])+1)]
# PlotData(x, y, "Wykres wzniesień", xx, yy)
Testy()


