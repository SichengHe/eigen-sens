# Sicheng He adapted from Matlab code from S. Peter, Dan. Henningson, "Stability and transition in shear flows"

import numpy as np
import scipy as sp
import scipy.linalg
import copy

def Dmat(N):

    '''
        Function to create differentiation matrices.

        N          = number of modes
        D0         = zero'th derivative matrix
        D1         = first derivative matrix
        D2         = second derivative matrix
        D4         = fourth derivative matrix
    '''

    # ========================
    # create D0
    # ========================
    # sampled over Gauss-Lobatto points: yj = cos(j pi / N)
    # for the n th mode, Tn(yj) = cos(n cos^-1 (y_j)) = cos(n j pi / N)
    D0 = np.zeros((N + 1, N + 1))
    vec = np.linspace(0, N, N + 1)
    for j in range(N + 1):
        D0[:, j] = np.cos(j * np.pi * vec / N)

    # ========================
    # create higher derivative matrices
    # ========================
    D1 = np.zeros_like(D0)
    D2 = np.zeros_like(D0)
    D3 = np.zeros_like(D0)
    D4 = np.zeros_like(D0)

    # T0^(k) = 0, 
    # T1^(k) = T0^(k-1)
    # T2^(k) = 4 T1^(k-1)
    D1[:, 1] = D0[:, 0]
    D1[:, 2] = 4.0 * D0[:, 1]
    D2[:, 2] = 4.0 * D0[:, 0]

    # Tn^(k) = 2 n * T(n-1)^(k-1) + n / (n - 2) * T(n-2)^(k)
    for j in range(3, N + 1):
        D1[:, j] = (2.0 * j) * D0[:, j - 1] + j * D1[:, j - 2] / (j - 2.0)
        D2[:, j] = (2.0 * j) * D1[:, j - 1] + j * D2[:, j - 2] / (j - 2.0)
        D3[:, j] = (2.0 * j) * D2[:, j - 1] + j * D3[:, j - 2] / (j - 2.0)
        D4[:, j] = (2.0 * j) * D3[:, j - 1] + j * D4[:, j - 2] / (j - 2.0)

    return D0, D1, D2, D4

def pois(nosmod, alp, beta, R, D0, D1, D2, D4):

    '''
        nosmod     = number of modes
        alp        = alpha
        beta       = beta
        R          = Renoyds number
        D0         = zero'th derivative matrix
        D1         = first derivative matrix
        D2         = second derivative matrix
        D4         = fourth derivative matrix
    '''

    zi = 1j
    SCALING_FACTOR = - 200.0 * zi 

    # ========================
    # mean velocity
    # ========================
    ak2 = alp ** 2 + beta ** 2
    Nos = nosmod + 1
    Nsq = nosmod + 1
    vec = np.linspace(0, nosmod, nosmod + 1)
    u = np.ones(len(vec)) - np.square(np.cos(np.pi * vec / nosmod))
    du = -2.0 * np.cos(np.pi * vec / nosmod)

    # ========================
    # set up Orr-Sommerfeld matrix
    # ========================
    B11 = D2 - ak2 * D0
    A11 = - (D4 - 2.0 * ak2 * D2 + (ak2 ** 2) * D0) / (zi * R)
    A11 = A11 + alp * np.outer(u, np.ones(len(u))) * B11 + alp * 2 * D0 # 2 is 2nd der of U

    # BC
    er = SCALING_FACTOR
    A11[0, :] = er * D0[0, :]
    A11[1, :] = er * D1[0, :]
    A11[-2, :] = er * D1[Nos - 1, :]
    A11[-1, :] = er * D0[Nos - 1, :]
    B11[0, :] = D0[0, :]
    B11[1, :] = D1[0, :]
    B11[-2, :] = D1[Nos - 1, :]
    B11[-1, :] = D0[Nos - 1, :]

    # set up Squire matrix and cross-term matrix
    A21 = beta * np.outer(du, np.ones(len(u))) * D0
    A22 = alp * np.outer(u, np.ones(len(u))) * D0 - (D2 - ak2 * D0) / (zi * R)
    B22 = D0
    A22[0, :] = er * D0[0, :]
    A22[-1, :] = er * D0[Nsq - 1, :]
    A21[0, :] *= 0
    A21[-1, :] *= 0

    # combine all blocks
    A = np.zeros((Nos + Nsq, Nos + Nsq), dtype = "D")
    A[0 : Nos, 0 : Nos] = A11[:, :]
    A[Nos : Nos + Nsq, 0 : Nos] = A21[:, :]
    A[Nos : Nos + Nsq, Nos : Nos + Nsq] = A22[:, :]
    
    B = np.zeros((Nos + Nsq, Nos + Nsq), dtype = "D")
    B[0 : Nos, 0 : Nos] = B11[:, :]
    B[Nos : Nos + Nsq, Nos : Nos + Nsq] = B22[:, :]

    return A, B

def extract_eig(w, vr):

    '''
        Extract the eigenpair with largest real part.
    '''

    N = len(w)

    eig_max = float('-inf') 
    ind = 0

    for i in range(N):
        if np.imag(w[i]) > eig_max:
            eig_max = np.imag(w[i])

            ind = i

    return w[ind], vr[:, ind]

def rotate_phi0(phi0):
    '''
        Rotate phi0 such that the largest entry aligned with the real axis.
    '''

    ind_max = 0 
    value_max = float('-inf') 

    N = len(phi0)

    for i in range(N):
        if np.linalg.norm(phi0[i]) > value_max:
            value_max = np.linalg.norm(phi0[i])

            ind_max = i

    theta = np.angle(phi0[ind_max])
    rot = np.exp(- 1j * theta)

    phi0_rotated = phi0 * rot

    return phi0_rotated

# ========================
# analysis
# ========================
R = 10000.0
alp = 1.0
beta = 0.0

N = 200
D0, D1, D2, D4 = Dmat(N)
K, M = pois(N, alp, beta, R, D0, D1, D2, D4)
# w, vr = sp.linalg.eig(A, b = B)
if 0:
    A = np.linalg.inv(M).dot(K)
    w, vr = sp.linalg.eig(A)
else:
    # M = np.eye(K.shape[0])
    w, vr = sp.linalg.eig(K, b = M)

print("w", w)
NN = K.shape[0]

lambda0, phi0 = extract_eig(w, vr)
phi0 = rotate_phi0(phi0)

# ========================
# adjoint
# ========================
def get_JT(A, lambda0, phi0):

    '''
        Form the transpose Jacobian matrix for the complex eigenvalue problem.
    '''


    # extract real and imag components
    phi0_r = np.real(phi0)
    phi0_i = np.imag(phi0)
    lambda0_r = np.real(lambda0)
    lambda0_i = np.imag(lambda0)
    A_r = np.real(A)
    A_i = np.imag(A)

    N = A.shape[0]
    I = np.eye(N)

    # figure out the biggest entry
    phi0_entry_norm = np.zeros_like(phi0_r)
    max_ind = 0
    max_value = -1.0
    for i in range(N):
        phi0_entry_norm[i] = np.linalg.norm(phi0[i])
        if phi0_entry_norm[i] > max_value:
            max_ind = i
            max_value = phi0_entry_norm[i]

    ek = np.zeros(N)
    ek[max_ind] = 1.0

    
    # form Jacobian
    pRpwT = np.zeros((2 * N + 2, 2 * N + 2))
    pRpwT[0:N, 0:N] = A_r[0:N, 0:N] - I * lambda0_r
    pRpwT[0:N, N:2 * N] = - A_i[0:N, 0:N] + I * lambda0_i
    pRpwT[N:2 * N, 0:N] = A_i[0:N, 0:N] - I * lambda0_i
    pRpwT[N:2 * N, N:2 * N] = A_r[0:N, 0:N] - I * lambda0_r
    pRpwT[0:N, 2 * N] = - phi0_r[:]
    pRpwT[0:N, 2 * N + 1] = phi0_i[:]
    pRpwT[N:2 * N, 2 * N] = - phi0_i[:]
    pRpwT[N:2 * N, 2 * N + 1] = - phi0_r[:]
    pRpwT[2 * N, 0:N] = 2 * phi0_r[:]
    pRpwT[2 * N, N:2 * N] = 2 * phi0_i[:]
    pRpwT[2 * N + 1, N:2 * N] = ek[:]
    pRpwT = pRpwT.T

    return pRpwT

def get_pfpw(c1, c2, N):

    '''
        Extract the RHS of the adjoint equation.
        Assume it takes the form:
             c1^T phi0 + c2 lambda0.
    '''

    c1_r = np.real(c1)
    c1_i = np.imag(c1)
    c2_r = np.real(c2)
    c2_i = np.imag(c2)

    # real RHS
    pfrpw = np.zeros(2 * N + 2)
    pfrpw[0:N] = c1_r[:]
    pfrpw[N:2 * N] = - c1_i[:]
    pfrpw[2 * N] = c2_r
    pfrpw[2 * N + 1] = - c2_i

    # complex RHS
    pfipw = np.zeros(2 * N + 2)
    pfipw[0:N] = c1_i[:]
    pfipw[N:2 * N] = c1_r[:]
    pfipw[2 * N] = c2_i
    pfipw[2 * N + 1] = c2_r 

    return pfrpw, pfipw

def compute_adjoint(A, lambda0, phi0, c1, c2):

    '''
        Solve the adjoint vectors for complex eigenvalue problem.
    '''

    pRpwT = get_JT(A, lambda0, phi0)

    N = A.shape[0]

    pfrpw, pfipw = get_pfpw(c1, c2, N)

    psi_fr = np.linalg.solve(pRpwT, pfrpw)
    psi_fi = np.linalg.solve(pRpwT, pfipw)

    return psi_fr, psi_fi

def compute_dfdA(psi_fr, psi_fi, phi0, N):

    '''
        Compute derivative matrices.
    '''

    psi_fr_main_r = copy.deepcopy(psi_fr[0:N])
    psi_fr_main_i = copy.deepcopy(psi_fr[N:2 * N])
    psi_fi_main_r = copy.deepcopy(psi_fi[0:N])
    psi_fi_main_i = copy.deepcopy(psi_fi[N:2 * N])
    
    phi0_r = np.real(phi0)
    phi0_i = np.imag(phi0)

    bar_fr_Ar = np.outer(psi_fr_main_r, phi0_r) + np.outer(psi_fr_main_i, phi0_i)
    bar_fr_Ai = - np.outer(psi_fr_main_r, phi0_i) + np.outer(psi_fr_main_i, phi0_r)
    bar_fi_Ar = np.outer(psi_fi_main_r, phi0_r) + np.outer(psi_fi_main_i, phi0_i)
    bar_fi_Ai = - np.outer(psi_fi_main_r, phi0_i) + np.outer(psi_fi_main_i, phi0_r)

    dfr_dAr = - bar_fr_Ar
    dfr_dAi = - bar_fr_Ai
    dfi_dAr = - bar_fi_Ar
    dfi_dAi = - bar_fi_Ai

    return dfr_dAr, dfr_dAi, dfi_dAr, dfi_dAi

def obj(c1, c2, phi0, lambda0):

    '''
        Compute the objective function.
    '''
    
    return c1.dot(phi0) + c2 * lambda0


def get_JT_gen(K, M, lambda0, phi0):

    '''
        Form the transpose Jacobian matrix for the complex *generalized* eigenvalue problem.
    '''


    # extract real and imag components
    phi0_r = np.real(phi0)
    phi0_i = np.imag(phi0)
    lambda0_r = np.real(lambda0)
    lambda0_i = np.imag(lambda0)
    K_r = np.real(K)
    K_i = np.imag(K)
    M_r = np.real(M)
    M_i = np.imag(M)

    N = K.shape[0]

    # figure out the biggest entry
    phi0_entry_norm = np.zeros_like(phi0_r)
    max_ind = 0
    max_value = -1.0
    for i in range(N):
        phi0_entry_norm[i] = np.linalg.norm(phi0[i])
        if phi0_entry_norm[i] > max_value:
            max_ind = i
            max_value = phi0_entry_norm[i]

    ek = np.zeros(N)
    ek[max_ind] = 1.0

    
    # form Jacobian
    pRpwT = np.zeros((2 * N + 2, 2 * N + 2))
    pRpwT[0:N, 0:N] = K_r[0:N, 0:N] - lambda0_r * M_r + lambda0_i * M_i
    pRpwT[0:N, N:2 * N] = - K_i[0:N, 0:N] + lambda0_i * M_r + lambda0_r * M_i
    pRpwT[N:2 * N, 0:N] = K_i[0:N, 0:N] - lambda0_i * M_r - lambda0_r * M_i
    pRpwT[N:2 * N, N:2 * N] = K_r[0:N, 0:N] - lambda0_r * M_r + lambda0_i * M_i
    pRpwT[0:N, 2 * N] = - M_r.dot(phi0_r) + M_i.dot(phi0_i)
    pRpwT[0:N, 2 * N + 1] = M_i.dot(phi0_r) + M_r.dot(phi0_i)
    pRpwT[N:2 * N, 2 * N] = - M_i.dot(phi0_r) - M_r.dot(phi0_i)
    pRpwT[N:2 * N, 2 * N + 1] = - M_r.dot(phi0_r) + M_i.dot(phi0_i)
    pRpwT[2 * N, 0:N] = 2 * phi0_r[:]
    pRpwT[2 * N, N:2 * N] = 2 * phi0_i[:]
    pRpwT[2 * N + 1, N:2 * N] = ek[:]
    pRpwT = pRpwT.T

    return pRpwT

def compute_adjoint_gen(K, M, lambda0, phi0, c1, c2):

    '''
        Solve the adjoint vectors for complex *generalized* eigenvalue problem.
    '''

    pRpwT = get_JT_gen(K, M, lambda0, phi0)

    N = K.shape[0]

    pfrpw, pfipw = get_pfpw(c1, c2, N)

    psi_fr = np.linalg.solve(pRpwT, pfrpw)
    psi_fi = np.linalg.solve(pRpwT, pfipw)

    return psi_fr, psi_fi

def compute_dfdA_gen(psi_fr, psi_fi, lambda0, phi0, N):

    '''
        Compute derivative matrices for the *generalized* eigenvalue problem.
    '''

    psi_fr_main_r = copy.deepcopy(psi_fr[0:N])
    psi_fr_main_i = copy.deepcopy(psi_fr[N:2 * N])
    psi_fi_main_r = copy.deepcopy(psi_fi[0:N])
    psi_fi_main_i = copy.deepcopy(psi_fi[N:2 * N])
    
    lambda0_r = np.real(lambda0)
    lambda0_i = np.imag(lambda0)

    phi0_r = np.real(phi0)
    phi0_i = np.imag(phi0)

    bar_fr_Kr = np.outer(psi_fr_main_r, phi0_r) + np.outer(psi_fr_main_i, phi0_i)
    bar_fr_Ki = - np.outer(psi_fr_main_r, phi0_i) + np.outer(psi_fr_main_i, phi0_r)
    bar_fi_Kr = np.outer(psi_fi_main_r, phi0_r) + np.outer(psi_fi_main_i, phi0_i)
    bar_fi_Ki = - np.outer(psi_fi_main_r, phi0_i) + np.outer(psi_fi_main_i, phi0_r)

    bar_fr_Mr = np.outer(psi_fr_main_r, - lambda0_r * phi0_r + lambda0_i * phi0_i) + np.outer(psi_fr_main_i, - lambda0_i * phi0_r - lambda0_r * phi0_i)
    bar_fr_Mi = np.outer(psi_fr_main_r, lambda0_i * phi0_r + lambda0_r * phi0_i) + np.outer(psi_fr_main_i, - lambda0_r * phi0_r + lambda0_i * phi0_i)
    bar_fi_Mr = np.outer(psi_fi_main_r, - lambda0_r * phi0_r + lambda0_i * phi0_i) + np.outer(psi_fi_main_i, - lambda0_i * phi0_r - lambda0_r * phi0_i)
    bar_fi_Mi = np.outer(psi_fi_main_r, lambda0_i * phi0_r + lambda0_r * phi0_i) + np.outer(psi_fi_main_i, - lambda0_r * phi0_r + lambda0_i * phi0_i)

    dfr_dKr = - bar_fr_Kr
    dfr_dKi = - bar_fr_Ki
    dfi_dKr = - bar_fi_Kr
    dfi_dKi = - bar_fi_Ki

    dfr_dMr = - bar_fr_Mr
    dfr_dMi = - bar_fr_Mi
    dfi_dMr = - bar_fi_Mr
    dfi_dMi = - bar_fi_Mi

    return dfr_dKr, dfr_dKi, dfi_dKr, dfi_dKi, dfr_dMr, dfr_dMi, dfi_dMr, dfi_dMi


c1 = np.ones(NN, dtype = 'D') * (1 + 1j)
c2 = 1.0

if 0:
    psi_fr, psi_fi = compute_adjoint(A, lambda0, phi0, c1, c2)
    dfr_dAr, dfr_dAi, dfi_dAr, dfi_dAi = compute_dfdA(psi_fr, psi_fi, phi0, NN)

psi_fr, psi_fi = compute_adjoint_gen(K, M, lambda0, phi0, c1, c2)
dfr_dKr, dfr_dKi, dfi_dKr, dfi_dKi, dfr_dMr, dfr_dMi, dfi_dMr, dfi_dMi = compute_dfdA_gen(psi_fr, psi_fi, lambda0, phi0, NN)


# ========================
# FD
# ========================

if 0:
    index_list = []
    N_ind = 10
    for i in range(N_ind):
        row_ind = 0
        col_ind = i

        index_list.append([row_ind, col_ind])

    obj_0 = obj(c1, c2, phi0, lambda0)
 
    epsilon = 1e-6

    # d obj_r / d A_r, d obj_i / d A_r
    dobj_dAr = []
    for i in range(len(index_list)):

        row_ind, col_ind = index_list[i]

        A_perturb = copy.deepcopy(A)
        A_perturb[row_ind, col_ind] += epsilon

        w_perturb, vr_perturb = sp.linalg.eig(A_perturb)
        lambda0_perturb, phi0_perturb = extract_eig(w_perturb, vr_perturb)

        obj_new = obj(c1, c2, phi0_perturb, lambda0_perturb)

        dobj_dAr.append((obj_new - obj_0) / epsilon)


    # d obj_r / d A_i, d obj_i / d A_i
    dobj_dAi = []
    for i in range(len(index_list)):

        row_ind, col_ind = index_list[i]

        A_perturb = copy.deepcopy(A)
        A_perturb[row_ind, col_ind] += epsilon * 1j

        w_perturb, vr_perturb = sp.linalg.eig(A_perturb)
        lambda0_perturb, phi0_perturb = extract_eig(w_perturb, vr_perturb)

        obj_new = obj(c1, c2, phi0_perturb, lambda0_perturb)

        dobj_dAi.append((obj_new - obj_0) / epsilon)


    df_dAr_adjoint_list = []
    df_dAi_adjoint_list = []
    df_dAr_FD_list = []
    df_dAi_FD_list = []

    for i in range(len(index_list)):

        row_ind, col_ind = index_list[i]

        df_dAr_adjoint = dfr_dAr[row_ind, col_ind] + 1j * dfi_dAr[row_ind, col_ind]
        df_dAi_adjoint = dfr_dAi[row_ind, col_ind] + 1j * dfi_dAi[row_ind, col_ind]

        df_dAr_FD = dobj_dAr[i]
        df_dAi_FD = dobj_dAi[i]

        df_dAr_adjoint_list.append(df_dAr_adjoint)
        df_dAi_adjoint_list.append(df_dAi_adjoint)

        df_dAr_FD_list.append(df_dAr_FD)
        df_dAi_FD_list.append(df_dAi_FD)

    print("d f / d Ar:")
    print("="*10)
    print("adjoint  |  FD")
    for i in range(len(index_list)):

        print(df_dAr_adjoint_list[i], df_dAr_FD_list[i])
        
    print("d f / d Ai:")
    print("="*10)
    print("adjoint  |  FD")
    for i in range(len(index_list)):

        print(df_dAi_adjoint_list[i], df_dAi_FD_list[i])    



index_list = []
N_ind = 5
for i in range(N_ind):
    row_ind = 0
    col_ind = i

    index_list.append([row_ind, col_ind])

obj_0 = obj(c1, c2, phi0, lambda0)

epsilon = 1e-6

# d obj_r / d M_r, d obj_i / d M_r
dobj_dMr = []
for i in range(len(index_list)):

    row_ind, col_ind = index_list[i]

    M_perturb = copy.deepcopy(M)
    M_perturb[row_ind, col_ind] += epsilon

    w_perturb, vr_perturb = sp.linalg.eig(K, b = M_perturb)
    lambda0_perturb, phi0_perturb = extract_eig(w_perturb, vr_perturb)
    phi0_perturb = rotate_phi0(phi0_perturb)

    obj_new = obj(c1, c2, phi0_perturb, lambda0_perturb)

    dobj_dMr.append((obj_new - obj_0) / epsilon)


# d obj_r / d M_i, d obj_i / d M_i
dobj_dMi = []
for i in range(len(index_list)):

    row_ind, col_ind = index_list[i]

    M_perturb = copy.deepcopy(M)
    M_perturb[row_ind, col_ind] += epsilon * 1j

    w_perturb, vr_perturb = sp.linalg.eig(K, b = M_perturb)
    lambda0_perturb, phi0_perturb = extract_eig(w_perturb, vr_perturb)
    phi0_perturb = rotate_phi0(phi0_perturb)

    obj_new = obj(c1, c2, phi0_perturb, lambda0_perturb)

    dobj_dMi.append((obj_new - obj_0) / epsilon)

# d obj_r / d K_r, d obj_i / d K_r
dobj_dKr = []
for i in range(len(index_list)):

    row_ind, col_ind = index_list[i]

    K_perturb = copy.deepcopy(K)
    K_perturb[row_ind, col_ind] += epsilon

    w_perturb, vr_perturb = sp.linalg.eig(K_perturb, b = M)
    lambda0_perturb, phi0_perturb = extract_eig(w_perturb, vr_perturb)
    phi0_perturb = rotate_phi0(phi0_perturb)

    obj_new = obj(c1, c2, phi0_perturb, lambda0_perturb)

    dobj_dKr.append((obj_new - obj_0) / epsilon)


# d obj_r / d K_i, d obj_i / d K_i
dobj_dKi = []
for i in range(len(index_list)):

    row_ind, col_ind = index_list[i]

    K_perturb = copy.deepcopy(K)
    K_perturb[row_ind, col_ind] += epsilon * 1j

    w_perturb, vr_perturb = sp.linalg.eig(K_perturb, b = M)
    lambda0_perturb, phi0_perturb = extract_eig(w_perturb, vr_perturb)
    phi0_perturb = rotate_phi0(phi0_perturb)

    obj_new = obj(c1, c2, phi0_perturb, lambda0_perturb)

    dobj_dKi.append((obj_new - obj_0) / epsilon)

df_dMr_adjoint_list = []
df_dMi_adjoint_list = []
df_dMr_FD_list = []
df_dMi_FD_list = []
df_dKr_adjoint_list = []
df_dKi_adjoint_list = []
df_dKr_FD_list = []
df_dKi_FD_list = []

for i in range(len(index_list)):

    row_ind, col_ind = index_list[i]

    df_dMr_adjoint = dfr_dMr[row_ind, col_ind] + 1j * dfi_dMr[row_ind, col_ind]
    df_dMi_adjoint = dfr_dMi[row_ind, col_ind] + 1j * dfi_dMi[row_ind, col_ind]
    df_dKr_adjoint = dfr_dKr[row_ind, col_ind] + 1j * dfi_dKr[row_ind, col_ind]
    df_dKi_adjoint = dfr_dKi[row_ind, col_ind] + 1j * dfi_dKi[row_ind, col_ind]

    df_dMr_FD = dobj_dMr[i]
    df_dMi_FD = dobj_dMi[i]
    df_dKr_FD = dobj_dKr[i]
    df_dKi_FD = dobj_dKi[i]

    df_dMr_adjoint_list.append(df_dMr_adjoint)
    df_dMi_adjoint_list.append(df_dMi_adjoint)
    df_dKr_adjoint_list.append(df_dKr_adjoint)
    df_dKi_adjoint_list.append(df_dKi_adjoint)

    df_dMr_FD_list.append(df_dMr_FD)
    df_dMi_FD_list.append(df_dMi_FD)
    df_dKr_FD_list.append(df_dKr_FD)
    df_dKi_FD_list.append(df_dKi_FD)

print("="*10)
print("d f / d Mr:")
print("-"*10)
print("adjoint  |  FD")
for i in range(len(index_list)):

    print(df_dMr_adjoint_list[i], df_dMr_FD_list[i])

print("="*10) 
print("d f / d Mi:")
print("-"*10)
print("adjoint  |  FD")
for i in range(len(index_list)):

    print(df_dMi_adjoint_list[i], df_dMi_FD_list[i])    

print("="*10)
print("d f / d Kr:")
print("-"*10)
print("adjoint  |  FD")
for i in range(len(index_list)):

    print(df_dKr_adjoint_list[i], df_dKr_FD_list[i])

print("="*10) 
print("d f / d Ki:")
print("-"*10)
print("adjoint  |  FD")
for i in range(len(index_list)):

    print(df_dKi_adjoint_list[i], df_dKi_FD_list[i])   


# ========================
# plot
# ========================
if 1:
    # plot eigenvalue distribution in complex plane
    import matplotlib.pyplot as plt
    from plot_utils import *

    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    w_r = np.real(w)
    w_i = np.imag(w)

    ax.plot([0, 1.1], [0, 0], '-', color='k', alpha=0.4, linewidth = 3)


    plt.plot(w_r, w_i, 'o', markersize = 8)
    plt.ylim(-1, 0.1)
    plt.xlim(0,1.1)
    ax.text(0.8, -0.3, r"P", color=my_blue, fontsize=20)
    ax.text(0.7, -0.6, r"S", color=my_blue, fontsize=20)
    ax.text(0.3, -0.3, r"A", color=my_blue, fontsize=20)

    ax.set_xlabel(r'$\omega_r$', fontsize=20, rotation=0)
    ax.set_ylabel(r'$\omega_i$', fontsize=20, rotation=0)



    plt.savefig('../R1_journal/figures/poiseuille_eig_2.pdf',bbox_inches='tight')
    plt.show()
