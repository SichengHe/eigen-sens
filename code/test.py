# Author: Sicheng He
# Date: 2020/01/10
# purpose: test cases for the paper
# "Complex matrix eigenvalue and eigenvector sensitivities with reverse automatic differentiation"
# Sicheng He, Eirikur Jonsson, Yayun Shi, Joaquim R. R. A. Martins

import numpy

numpy.set_printoptions(formatter={"float": lambda x: "{0:0.15f}".format(x)})


def f_func(phi, lam, phi_c, lam_c):

    # objective function
    # phi_c and lam_c are coeffients

    return numpy.dot(phi, phi_c) + lam * lam_c


isRAD = False

# ==================
# FAD and FD test
# ==================
N = 3

if not isRAD:
    phi_cr = numpy.array([0.16, 0.53, 0.11])
    phi_ci = numpy.array([0.78, 0.11, 0.77])
else:
    phi_cr = numpy.array([0, 0, 0])
    phi_ci = numpy.array([0, 0, 0])
phi_c = phi_cr + 1j * phi_ci
if not isRAD:
    lam_cr = 1.0
    lam_ci = 0.5
else:
    lam_cr = 1.0
    lam_ci = 0.0
lam_c = lam_cr + 1j * lam_ci

A_r = numpy.array([[-1.01, 0.86, -4.60], [3.98, 0.53, -7.04], [3.30, 8.26, -3.89]])

A_i = numpy.array([[0.30, 0.79, 5.47], [7.21, 1.90, 0.58], [3.42, 8.97, 0.30]])

Lambda0, Phi0 = numpy.linalg.eig(A_r + A_i * 1j)
phi0 = Phi0[:, 0]
lambda0 = Lambda0[0]

f0 = f_func(phi0, lambda0, phi_c, lam_c)

# ----------------
# FD
# ----------------

epsilon = 1e-6

dfdAr_FD = numpy.zeros((N, N), dtype="D")
dfdAi_FD = numpy.zeros((N, N), dtype="D")

deltaA_r = numpy.zeros((N, N))
deltaA_i = numpy.zeros((N, N))

for i in range(N):
    for j in range(N):
        # perturb
        deltaA_r[i, j] = 1.0
        Lambda1, Phi1 = numpy.linalg.eig(A_r + A_i * 1j + deltaA_r * epsilon + deltaA_i * epsilon * 1j)

        phi1 = Phi1[:, 0]
        lambda1 = Lambda1[0]

        f1 = f_func(phi1, lambda1, phi_c, lam_c)

        dfdx = (f1 - f0) / epsilon

        dfdAr_FD[i, j] = dfdx

        # reset
        deltaA_r[i, j] = 0.0

for i in range(N):
    for j in range(N):
        # perturb
        deltaA_i[i, j] = 1.0
        Lambda1, Phi1 = numpy.linalg.eig(A_r + A_i * 1j + deltaA_r * epsilon + deltaA_i * epsilon * 1j)

        phi1 = Phi1[:, 0]
        lambda1 = Lambda1[0]

        f1 = f_func(phi1, lambda1, phi_c, lam_c)

        dfdx = (f1 - f0) / epsilon

        dfdAi_FD[i, j] = dfdx

        # reset
        deltaA_i[i, j] = 0.0

print("dfrdAr_FD", numpy.real(dfdAr_FD))
print("dfidAr_FD", numpy.imag(dfdAr_FD))
print("dfrdAi_FD", numpy.real(dfdAi_FD))
print("dfidAi_FD", numpy.imag(dfdAi_FD))

# ----------------
# adjoint
# ----------------
phi0_r = numpy.real(phi0)
phi0_i = numpy.imag(phi0)
lambda0_r = numpy.real(lambda0)
lambda0_i = numpy.imag(lambda0)
I = numpy.eye(N)

print("phi0_r", phi0_r)
print("phi0_i", phi0_i)
print("lambda0_r", lambda0_r)
print("lambda0_i", lambda0_i)

phi0_entry_norm = numpy.zeros_like(phi0_r)
max_ind = 0
max_value = -1.0
for i in range(N):
    phi0_entry_norm[i] = numpy.linalg.norm(phi0[i])
    if phi0_entry_norm[i] > max_value:
        max_ind = i
        max_value = phi0_entry_norm[i]

ek = numpy.zeros(N)
ek[max_ind] = 1.0


pRpwT = numpy.zeros((2 * N + 2, 2 * N + 2))
pRpwT[0:N, 0:N] = A_r[0:N, 0:N] - I * lambda0_r
pRpwT[0:N, N : 2 * N] = -A_i[0:N, 0:N] + I * lambda0_i
pRpwT[N : 2 * N, 0:N] = A_i[0:N, 0:N] - I * lambda0_i
pRpwT[N : 2 * N, N : 2 * N] = A_r[0:N, 0:N] - I * lambda0_r
pRpwT[0:N, 2 * N] = -phi0_r[:]
pRpwT[0:N, 2 * N + 1] = phi0_i[:]
pRpwT[N : 2 * N, 2 * N] = -phi0_i[:]
pRpwT[N : 2 * N, 2 * N + 1] = -phi0_r[:]
pRpwT[2 * N, 0:N] = 2 * phi0_r[:]
pRpwT[2 * N, N : 2 * N] = 2 * phi0_i[:]
pRpwT[2 * N + 1, N : 2 * N] = ek[:]
pRpwT = pRpwT.T

# real RHS
dfrdw = numpy.zeros(2 * N + 2)
dfrdw[0:N] = phi_cr[:]
dfrdw[N : 2 * N] = -phi_ci[:]
dfrdw[2 * N] = lam_cr
dfrdw[2 * N + 1] = -lam_ci

# complex RHS
dfidw = numpy.zeros(2 * N + 2)
dfidw[0:N] = phi_ci[:]
dfidw[N : 2 * N] = phi_cr[:]
dfidw[2 * N] = lam_ci
dfidw[2 * N + 1] = lam_cr

psi_fr = numpy.linalg.solve(pRpwT, dfrdw)
psi_fr_main_r = psi_fr[0:N]
psi_fr_main_i = psi_fr[N : 2 * N]
psi_fr_m = psi_fr[2 * N]
psi_fr_p = psi_fr[2 * N + 1]

psi_fi = numpy.linalg.solve(pRpwT, dfidw)
psi_fi_main_r = psi_fi[0:N]
psi_fi_main_i = psi_fi[N : 2 * N]
psi_fi_m = psi_fi[2 * N]
psi_fi_p = psi_fi[2 * N + 1]

bar_fr_Ar = numpy.outer(psi_fr_main_r, phi0_r) + numpy.outer(psi_fr_main_i, phi0_i)
bar_fr_Ai = -numpy.outer(psi_fr_main_r, phi0_i) + numpy.outer(psi_fr_main_i, phi0_r)
bar_fi_Ar = numpy.outer(psi_fi_main_r, phi0_r) + numpy.outer(psi_fi_main_i, phi0_i)
bar_fi_Ai = -numpy.outer(psi_fi_main_r, phi0_i) + numpy.outer(psi_fi_main_i, phi0_r)


print("- bar_fr_Ar", -bar_fr_Ar)
print("- bar_fr_Ai", -bar_fr_Ai)
print("- bar_fi_Ar", -bar_fi_Ar)
print("- bar_fi_Ai", -bar_fi_Ai)

# ----------------
# RAD
# ----------------

if isRAD:
    # Compute the corresponding left eigenvector first
    index = 0
    Lambda0_left, Phi0_left = numpy.linalg.eig(A_r.T - A_i.T * 1j)
    phi0_left = Phi0_left[:, index]
    lambda0_left = Lambda0_left[index]

    print("Lambda0", Lambda0[0])
    print("Lambda0", Lambda0[1])
    print("Lambda0", Lambda0[2])

    mat = numpy.outer(phi0_left, numpy.conjugate(phi0)) / numpy.dot(phi0_left, numpy.conjugate(phi0))

    d_lambdar_Ar = numpy.real(mat)
    d_lambdar_Ai = numpy.imag(mat)
    d_lambdai_Ar = -d_lambdar_Ai
    d_lambdai_Ai = d_lambdar_Ar

    d_fr_Ar = d_lambdar_Ar
    d_fr_Ai = d_lambdar_Ai
    d_fi_Ar = d_lambdai_Ar
    d_fi_Ai = d_lambdai_Ai

    print("d_fr_Ar", d_fr_Ar)
    print("d_fr_Ai", d_fr_Ai)
    print("d_fi_Ar", d_fi_Ar)
    print("d_fi_Ai", d_fi_Ai)
