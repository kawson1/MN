from Functions import *
from random import randint
import numpy as np


x, y = LoadDataFromFile("./2018_paths/SpacerniakGdansk.csv")
N = 20

# PlotMetodaLagrange(x, y, N)
# PlotMetodaSklejania(x, y, N)

x = [[0,1,2,3],
     [2,1,3,4],
     [3,4,1,5],
     [6,7,8,1]]

y = [1,2,3,4]

print(LU(x, y))
print(linalg.solve(x, y))
