import numpy as np
from eigen import eigen

np.set_printoptions(formatter={"float": lambda x: "{0:0.15f}".format(x)})


def f_func(phi, lam, phi_c, lam_c):

    # objective function
    # phi_c and lam_c are coeffients

    return np.dot(phi, phi_c) + lam * lam_c


isRAD = False

# ==================
# FAD and FD test
# ==================
N = 3

if not isRAD:
    phi_cr = np.array([0.16, 0.53, 0.11])
    phi_ci = np.array([0.78, 0.11, 0.77])
else:
    phi_cr = np.array([0, 0, 0])
    phi_ci = np.array([0, 0, 0])
phi_c = phi_cr + 1j * phi_ci
if not isRAD:
    lam_cr = 1.0
    lam_ci = 0.5
else:
    lam_cr = 1.0
    lam_ci = 0.0
lam_c = lam_cr + 1j * lam_ci

A_r = np.array([[-1.01, 0.86, -4.60], [3.98, 0.53, -7.04], [3.30, 8.26, -3.89]])

A_i = np.array([[0.30, 0.79, 5.47], [7.21, 1.90, 0.58], [3.42, 8.97, 0.30]])

A = A_r + 1j * A_i


# real
pfpeig_val = np.zeros(2)
pfpeig_vec = np.zeros(2*N)

pfpeig_vec[0:N] = phi_cr
pfpeig_vec[N:2*N] = - phi_ci
pfpeig_val[0] = lam_cr
pfpeig_val[1] = - lam_ci

eigen_obj = eigen(A, pfpeig_val, pfpeig_vec)
eigen_obj.solve(eig_ind = 0)
eig_vec_r, eig_vec_i, eig_val_r, eig_val_i = eigen_obj.get_sol()
print("eig_vec_r, eig_vec_i, eig_val_r, eig_val_i", eig_vec_r, eig_vec_i, eig_val_r, eig_val_i)
eigen_obj.setup_adjoint()
eigen_obj.solve_adjoint()
eigen_obj.compute_total_der()
[dfdA_r, dfdA_i] = eigen_obj.get_total_der()
print("dfdA_r", dfdA_r) 
print("dfdA_i", dfdA_i)