import numpy as np
import scipy as sp
from poiseuille import poiseuille


R = 10000.0
alp = 1.0
beta = 0.0

N = 200
poiseuille_obj = poiseuille(N)
poiseuille_obj.setup(alp, beta, R)
eig_val, eig_vec = poiseuille_obj.solve(isFull = False)
print("eig_val", eig_val)
# eig_val [0.23752649+0.00373967j]
