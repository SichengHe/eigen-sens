import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse.linalg
import copy
import time
import matplotlib.pylab as plt

# TODOs:
# 1. Add RAD implementation.

# HACK:
# 1. When computing the eigenvalue der, we assume it is the dominant eigenvalue real part.-> suppose to be any one.

class eigen(object):

    def __init__(self, A, pfpeig_val, pfpeig_vec):

        '''
            Eigen problem solver and sensitivity solver. 
            Notice that the function is either pure real or pure imaginary number.
            If it is complex, then we need to solve the adjoint twice.
        '''

        self.A = A
        # plt.spy(A)
        # plt.show()
        self.A_r = np.real(A)
        self.A_i = np.imag(A)

        self.A_H = sp.sparse.csr_matrix(np.conj(A).T)

        self.N = A.shape[0]
        N = self.N

        self.pfpeig_val_r = pfpeig_val[0]
        self.pfpeig_val_i = pfpeig_val[1]

        self.pfpeig_vec_r = pfpeig_vec[0:N]
        self.pfpeig_vec_i = pfpeig_vec[N:2*N]

        self.case_dict = {0:"LM", 1:"SM", 2: "LR", 3:"SR", 4:"LI", 5:"SI"}

    def normalize_eig_vec(self, eig_vec):

        # Normalize the eigenvector such that ||eig_vec|| = 1, phi(eig_vec[ind_max]) = 0.
        # Notice that we DO NOT normalize the length explicitly since it was 
        # taken care in eigenvector computation already.

        ind_max = self.compute_ind_max(eig_vec)

        # Get its angle
        theta = np.angle(eig_vec[ind_max])
        rot = np.exp(- 1j * theta)

        # Rotate the eigenvector accordingly
        eig_vec = eig_vec * rot

        self.ind_max = ind_max

        return eig_vec

    def compute_ind_max(self, eig_vec):

        # Get the maximum modulus index
        abs_eig_vec = abs(eig_vec)
        ind_max = np.argmax(abs_eig_vec)

        return ind_max

    def get_ind_max(self):

        return self.ind_max

    def solve(self, eig_ind = 2):

        A = self.A
    
        code_string = self.case_dict[eig_ind]

        t1 = time.time()
        eig_val, eig_vec = sp.sparse.linalg.eigs(A, which = code_string, k = 1, tol = 1e-8)
        t2 = time.time()
        print("Eigen solution time:", t2 - t1)

        self.eig_val = eig_val
        self.eig_vec = eig_vec

        self.ind_max = self.compute_ind_max(self.eig_vec)
        self.eig_vec = self.normalize_eig_vec(self.eig_vec)

        self.eig_vec_r = np.real(self.eig_vec)
        self.eig_vec_i = np.imag(self.eig_vec)
        self.eig_val_r = np.real(self.eig_val)
        self.eig_val_i = np.imag(self.eig_val)

        return t2 - t1

    def get_sol(self):

        return self.eig_vec_r, self.eig_vec_i, self.eig_val_r, self.eig_val_i

    def setup_adjoint(self, coeff_mat_sparse = None):

        N = self.N

        A_r = self.A_r
        A_i = self.A_i
        eig_val_r = np.real(self.eig_val)
        eig_val_i = np.imag(self.eig_val)
        eig_vec_r = np.real(self.eig_vec)
        eig_vec_i = np.imag(self.eig_vec)
        I_N = np.eye(N)

        pfpeig_val_r = self.pfpeig_val_r
        pfpeig_val_i = self.pfpeig_val_i
        pfpeig_vec_r = self.pfpeig_vec_r
        pfpeig_vec_i = self.pfpeig_vec_i

        # Form the coefficient matrix
        ek = np.zeros(N)
        ek[self.ind_max] = 1.0

        if (coeff_mat_sparse is None):
            coeff_mat = np.zeros((2 * N + 2, 2 * N + 2))
            coeff_mat[0:N, 0:N] = A_r - eig_val_r * I_N
            coeff_mat[0:N, N:2*N] = - A_i + eig_val_i * I_N
            coeff_mat[N:2*N, 0:N] = A_i - eig_val_i * I_N
            coeff_mat[N:2*N, N:2*N] = A_r - eig_val_r * I_N
            coeff_mat[0:N, 2*N] = -eig_vec_r[:, 0]
            coeff_mat[0:N, 2*N+1] = eig_vec_i[:, 0]
            coeff_mat[N:2*N, 2*N] = -eig_vec_i[:, 0]
            coeff_mat[N:2*N, 2*N+1] = -eig_vec_r[:, 0]
            coeff_mat[2*N, 0:N] = 2 * eig_vec_r[:, 0]
            coeff_mat[2*N, N:2*N] = 2 * eig_vec_i[:, 0]
            coeff_mat[2*N+1, N:2*N] = ek[:]

            self.coeff_mat = coeff_mat.T

            self.coeff_mat_sparse = sp.sparse.csr_matrix(self.coeff_mat)
        else:
            self.coeff_mat_sparse = coeff_mat_sparse.T

        # Form the RHS
        rhs_vec = np.zeros(2 * N + 2)
        rhs_vec[0:N] = pfpeig_vec_r[:]
        rhs_vec[N:2*N] = pfpeig_vec_i[:]
        rhs_vec[2*N] = pfpeig_val_r
        rhs_vec[2*N+1] = pfpeig_val_i

        self.rhs_vec = rhs_vec

    def solve_adjoint(self, isSparse = False):

        coeff_mat_sparse = self.coeff_mat_sparse
        rhs_vec = self.rhs_vec

        N = self.N

        t1 = time.time()
        if isSparse:
            psi = sp.sparse.linalg.gmres(coeff_mat_sparse, rhs_vec, tol=1e-8)[0]
        else:
            psi = np.linalg.solve(self.coeff_mat, rhs_vec)
        t2 = time.time()
        print("Adjoint solution time:", t2 - t1)

        psi_eig_vec_r = np.zeros(N)
        psi_eig_vec_i = np.zeros(N)

        psi_eig_vec_r[:] = psi[0:N]
        psi_eig_vec_i[:] = psi[N:2*N]

        psi_eig_val_r = psi[2*N]
        psi_eig_val_i = psi[2*N+1]

        self.psi_eig_vec_r = psi_eig_vec_r
        self.psi_eig_vec_i = psi_eig_vec_i
        self.psi_eig_val_r = psi_eig_val_r
        self.psi_eig_val_i = psi_eig_val_i

        return t2 - t1

    def compute_total_der(self):

        psi_eig_vec_r = self.psi_eig_vec_r
        psi_eig_vec_i = self.psi_eig_vec_i

        eig_vec_r = self.eig_vec_r
        eig_vec_i = self.eig_vec_i

        t0 = time.time()
        A_r_bar = np.outer(psi_eig_vec_r, eig_vec_r) + np.outer(psi_eig_vec_i, eig_vec_i)
        A_i_bar = -np.outer(psi_eig_vec_r, eig_vec_i) + np.outer(psi_eig_vec_i, eig_vec_r)
        t1 = time.time()

        print("Final sol time: ", t1 - t0)

        self.dfdA_r = -A_r_bar
        self.dfdA_i = -A_i_bar

    def get_der(self):

        return self.dfdA_r, self.dfdA_i

    def compute_total_der_adjugate(self):

        A = self.A
        A_H = self.A_H
        N = self.N

        A_arr = A.toarray()
        A_H_arr = A_H.toarray()

        t0 = time.time()
        eig_vals, eig_vecs = sp.linalg.eig(A_arr)
        eig_vals_H, eig_vecs_H = sp.linalg.eig(A_H_arr)
        t1 = time.time()
        print("Adjugate solution time: ", t1 - t0)

        # Sorting array
        eig_vals_inds = eig_vals.argsort()
        eig_vals = eig_vals[eig_vals_inds[::-1]]
        eig_vecs = eig_vecs[eig_vals_inds[::-1]]

        eig_vals_H_inds = eig_vals_H.argsort()
        eig_vals_H = eig_vals_H[eig_vals_H_inds[::-1]]
        eig_vecs_H = eig_vecs_H[eig_vals_H_inds[::-1]]

        return t1 - t0

    def solve_left(self,  eig_ind = 2):

        # Solve the left eigenvalue problem

        A_H = self.A_H
        eig_val = self.eig_val
        eig_vec = self.eig_vec

        code_string = self.case_dict[eig_ind]

        t1 = time.time()
        eig_val_H, eig_vec_H = sp.sparse.linalg.eigs(A_H, k = 1, sigma = np.conj(eig_val[0]), tol = 1e-8)
        # eig_val_H, eig_vec_H = sp.sparse.linalg.eigs(A_H_sparse, which = code_string, k = 1)
        t2 = time.time()
        print("RAD time: ", t2 - t1)
        
        self.eig_val_H = eig_val_H
        self.eig_vec_H = eig_vec_H

        eig_vec_H = self.normalize_eig_vec(eig_vec_H)

        return t2 - t1

    def compute_total_der_rad(self, isReal = True):

        # Compute the total derivative of the eigenvalue wrt the real and imaginary parts of the matrix.
        # isReal: If set true, then compute the derivative of the real part of the eigenvalue; otherwise, the imaginary part.

        eig_val = self.eig_val
        eig_vec = self.eig_vec

        eig_val_H = self.eig_val_H
        eig_vec_H = self.eig_vec_H

        t0 = time.time()
        deig_val_dA = np.outer(eig_vec_H, np.conj(eig_vec)) / (np.conj(eig_vec).T.dot(eig_vec_H))
        t1 = time.time()
        print("Eigen final solution time: ", t1 - t0)

        if isReal:
            deig_val_dAr = np.real(deig_val_dA)
            deig_val_dAi = np.imag(deig_val_dA)
        else:
            deig_val_dAr = -np.imag(deig_val_dA)
            deig_val_dAi = np.real(deig_val_dA)
        

        self.dfdA_r = deig_val_dAr
        self.dfdA_i = deig_val_dAi

    def get_total_der(self):

        return [self.dfdA_r, self.dfdA_i]
