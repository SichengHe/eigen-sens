from platform import java_ver
from re import L
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from eigen import eigen
from poiseuille import poiseuille
import matplotlib.pylab as plt
from plot_utils import *
import time


isVerifying = False
isTraining = False

def run(N, seed = 0):

    # Set the seeds
    np.random.seed(seed)

    # ==========================
    # Generate the random matrix
    # ==========================
    NN = 3 * N - 2
    data_r = np.random.rand(NN)
    data_i = np.random.rand(NN)
    data = data_r + data_i * 1j

    row = np.zeros(NN)
    col = np.zeros(NN)

    # upper diag
    ind = 0
    for i in range(N - 1):
        row[ind] = i
        col[ind] = i + 1
        ind += 1
    # diag
    for i in range(N):
        row[ind] = i
        col[ind] = i
        ind += 1
    # lower
    for i in range(N - 1):
        row[ind] = i + 1
        col[ind] = i
        ind += 1
    
    # creating sparse matrix
    A = csr_matrix((data, (row, col)), 
                            shape = (N, N))

    # ==========================
    # Solve the prime
    # ==========================

    N = A.shape[0]

    # Set the RHS for the adjoint and set up the problem
    pfpeig_val = np.zeros(2)
    pfpeig_vec = np.random.rand(N * 2) # This decides the function of interest
    eigen_obj = eigen(A, pfpeig_val, pfpeig_vec)
    t_eig = eigen_obj.solve(eig_ind = 2)

    # ==========================
    # Solve the Adjoint
    # ==========================
    # Construct coefficient matrix for the adjoint equation
    eig_vec_r, eig_vec_i, eig_val_r, eig_val_i = eigen_obj.get_sol()
    ind_max = eigen_obj.get_ind_max()

    coeff_mat_sparse_data = np.zeros(NN * 4 + N * 6 + 1)
    coeff_mat_sparse_row = np.zeros(NN * 4 + N * 6 + 1)
    coeff_mat_sparse_col = np.zeros(NN * 4 + N * 6 + 1)

    # --------------------------
    # (1, 1) Ar - lambdar I
    # --------------------------
    col_shift = 0
    row_shift = 0
    ind_shift = 0

    # upper diag
    ind_loc = 0
    for i in range(N - 1):

        # Local matrix iterator index
        i_loc = i
        j_loc = i + 1

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_r[ind_loc]

        ind_loc += 1

    # diag
    for i in range(N):
        
        # Local matrix iterator index
        i_loc = i
        j_loc = i

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_r[ind_loc] - eig_val_r

        ind_loc += 1

    # lower
    for i in range(N - 1):

        # Local matrix iterator index
        i_loc = i + 1
        j_loc = i

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_r[ind_loc]

        ind_loc += 1
    
    # --------------------------
    # (1, 2) -Ai + lambdai I
    # --------------------------
    col_shift = N
    row_shift = 0
    ind_shift = NN
    
    # upper diag
    ind_loc = 0
    for i in range(N - 1):

        # Local matrix iterator index
        i_loc = i
        j_loc = i + 1

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = -data_i[ind_loc]

        ind_loc += 1

    # diag
    for i in range(N):
        
        # Local matrix iterator index
        i_loc = i
        j_loc = i

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = -data_i[ind_loc] + eig_val_i

        ind_loc += 1

    # lower
    for i in range(N - 1):

        # Local matrix iterator index
        i_loc = i + 1
        j_loc = i

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = -data_i[ind_loc]

        ind_loc += 1

    # --------------------------
    # (2, 1) Ai - lambdai I
    # --------------------------
    col_shift = 0
    row_shift = N
    ind_shift = 2 * NN
    
    # upper diag
    ind_loc = 0
    for i in range(N - 1):

        # Local matrix iterator index
        i_loc = i
        j_loc = i + 1

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_i[ind_loc]

        ind_loc += 1

    # diag
    for i in range(N):
        
        # Local matrix iterator index
        i_loc = i
        j_loc = i

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_i[ind_loc] - eig_val_i

        ind_loc += 1

    # lower
    for i in range(N - 1):

        # Local matrix iterator index
        i_loc = i + 1
        j_loc = i

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_i[ind_loc]

        ind_loc += 1

    # --------------------------
    # (2, 2) Ar - lambdar I
    # --------------------------
    col_shift = N
    row_shift = N
    ind_shift = 3 * NN
    
    # upper diag
    ind_loc = 0
    for i in range(N - 1):

        # Local matrix iterator index
        i_loc = i
        j_loc = i + 1

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_r[ind_loc]

        ind_loc += 1

    # diag
    for i in range(N):
        
        # Local matrix iterator index
        i_loc = i
        j_loc = i

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_r[ind_loc] - eig_val_r

        ind_loc += 1

    # lower
    for i in range(N - 1):

        # Local matrix iterator index
        i_loc = i + 1
        j_loc = i

        # Global matrix iterator index
        i_glo = i_loc + row_shift
        j_glo = j_loc + col_shift
        ind_glo = ind_loc + ind_shift

        coeff_mat_sparse_row[ind_glo] = i_glo
        coeff_mat_sparse_col[ind_glo] = j_glo
        coeff_mat_sparse_data[ind_glo] = data_r[ind_loc]

        ind_loc += 1

    # --------------------------
    # Rest columns
    # --------------------------
    for i in range(N):

        coeff_mat_sparse_row[ind_glo] = 2 * N
        coeff_mat_sparse_col[ind_glo] = i
        coeff_mat_sparse_data[ind_glo] = 2 * eig_vec_r[i]

        ind_glo += 1

    for i in range(N):
    
        coeff_mat_sparse_row[ind_glo] = 2 * N
        coeff_mat_sparse_col[ind_glo] = N + i
        coeff_mat_sparse_data[ind_glo] = 2 * eig_vec_i[i]

        ind_glo += 1
    
    for i in range(N):
        
        coeff_mat_sparse_row[ind_glo] = i
        coeff_mat_sparse_col[ind_glo] = 2 * N
        coeff_mat_sparse_data[ind_glo] = - eig_vec_r[i]

        ind_glo += 1

    for i in range(N):
        
        coeff_mat_sparse_row[ind_glo] = N + i
        coeff_mat_sparse_col[ind_glo] = 2 * N
        coeff_mat_sparse_data[ind_glo] = - eig_vec_i[i]

        ind_glo += 1

    for i in range(N):
        
        coeff_mat_sparse_row[ind_glo] = i
        coeff_mat_sparse_col[ind_glo] = 2 * N + 1
        coeff_mat_sparse_data[ind_glo] = eig_vec_i[i]

        ind_glo += 1

    for i in range(N):
        
        coeff_mat_sparse_row[ind_glo] = N + i
        coeff_mat_sparse_col[ind_glo] = 2 * N + 1
        coeff_mat_sparse_data[ind_glo] = - eig_vec_r[i]

        ind_glo += 1

    # --------------------------
    # Entry with ind_max
    # --------------------------
    coeff_mat_sparse_row[ind_glo] = 2 * N + 1
    coeff_mat_sparse_col[ind_glo] = N + ind_max
    coeff_mat_sparse_data[ind_glo] = 1.0


    coeff_mat_sparse_data = csr_matrix((coeff_mat_sparse_data, (coeff_mat_sparse_row, coeff_mat_sparse_col)), 
                            shape = (2 * N + 2, 2 * N + 2))

    eigen_obj.setup_adjoint(coeff_mat_sparse = coeff_mat_sparse_data)
    t_adjoint_sparse = eigen_obj.solve_adjoint(isSparse = True)
    if isVerifying:
        eigen_obj.compute_total_der()
        deig_val_dAr_adjoint, deig_val_dAi_adjoint = eigen_obj.get_der()

    # ==========================
    # adjugate
    # ==========================

    t_adjugate = eigen_obj.compute_total_der_adjugate()
    
    # ==========================
    # Compare solution
    # ==========================
    if isVerifying:

        deig_val_dAr_adjoint_norm = np.linalg.norm(deig_val_dAr_adjoint, ord = "fro")
        deig_val_dAr_RAD_norm = np.linalg.norm(deig_val_dAr_RAD, ord = "fro")
        deig_val_dAr_err = np.linalg.norm(deig_val_dAr_adjoint - deig_val_dAr_RAD, ord = "fro")
        deig_val_dAr_rel_err = deig_val_dAr_err / deig_val_dAr_adjoint_norm
        

        deig_val_dAi_adjoint_norm = np.linalg.norm(deig_val_dAi_adjoint, ord = "fro")
        deig_val_dAi_RAD_norm = np.linalg.norm(deig_val_dAi_RAD, ord = "fro")
        deig_val_dAi_err = np.linalg.norm(deig_val_dAi_adjoint - deig_val_dAi_RAD, ord = "fro")
        deig_val_dAi_rel_err = deig_val_dAi_err / deig_val_dAi_adjoint_norm


        print("deig_val_dAr_adjoint_norm", deig_val_dAr_adjoint_norm, "deig_val_dAr_RAD_norm", deig_val_dAr_RAD_norm, "deig_val_dAr_err", deig_val_dAr_err, "deig_val_dAr_rel_err", deig_val_dAr_rel_err)
        print("deig_val_dAi_adjoint_norm", deig_val_dAi_adjoint_norm, "deig_val_dAi_RAD_norm", deig_val_dAi_RAD_norm, "deig_val_dAi_err", deig_val_dAi_err, "deig_val_dAi_rel_err", deig_val_dAi_rel_err)

    return t_eig, t_adjoint_sparse, t_adjugate

N_arr = np.array([128, 256, 512, 1024, 1024 * 2])
# N_arr = np.array([8192, 8192 * 2, 8192 * 4, 8192 * 8, 8192 * 16, 8192 * 32, 8192 * 64, 8192 * 128])

# seed_arr = np.array([0])
seed_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

n_pts = N_arr.shape[0]
n_test = seed_arr.shape[0]

if isTraining:
    for ii in range(n_test):

        t_array = np.zeros((n_pts, 3))

        i = 0
        for N in N_arr:
            t_eig, t_adjoint_sparse, t_adjugate = run(N, seed = i)

            t_array[i, 0] = t_eig
            t_array[i, 1] = t_adjoint_sparse
            t_array[i, 2] = t_adjugate

            print("t_eig, t_adjoint_sparse, t_adjugate", t_eig, t_adjoint_sparse, t_adjugate)

            i += 1

        np.savetxt("adjugate_adjoint_scaling_" + str(ii), t_array)

else:
    
    # load data
    data = np.zeros((n_pts, 3, n_test))
    for i in range(n_test):
        name = "adjugate_adjoint_scaling_" + str(i)
        data[:, :, i] = np.loadtxt(name)[:, :]

    t_eig_arr = np.zeros((n_pts, 2))
    t_adjoint_arr = np.zeros((n_pts, 2))
    t_adjugate_arr = np.zeros((n_pts, 2))

    for i in range(n_pts):
        
        t_eig_data_i = data[i, 0, :]
        t_adjoint_data_i = data[i, 1, :]
        t_adjugate_data_i = data[i, 2, :]

        t_eig_mean = np.average(t_eig_data_i)
        t_eig_var = np.sqrt(np.mean((t_eig_data_i - t_eig_mean) ** 2))
        t_eig_arr[i, 0] = t_eig_mean
        t_eig_arr[i, 1] = t_eig_var


        t_adjoint_mean = np.average(t_adjoint_data_i)
        t_adjoint_var = np.sqrt(np.mean((t_adjoint_data_i - t_adjoint_mean) ** 2))
        t_adjoint_arr[i, 0] = t_adjoint_mean
        t_adjoint_arr[i, 1] = t_adjoint_var

        t_adjugate_mean = np.average(t_adjugate_data_i)
        t_adjugate_var = np.sqrt(np.mean((t_adjugate_data_i - t_adjugate_mean) ** 2))
        t_adjugate_arr[i, 0] = t_adjugate_mean
        t_adjugate_arr[i, 1] = t_adjugate_var

    print(t_eig_arr, t_adjoint_arr, t_adjugate_arr)

    fig, ax = plt.subplots(1, 1, figsize=(10,6))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    print(t_eig_arr[:, 1])
    ax.errorbar(N_arr, t_eig_arr[:, 0], yerr = t_eig_arr[:, 1], color = my_blue)
    ax.errorbar(N_arr, t_adjoint_arr[:, 0], yerr = t_adjoint_arr[:, 1], color = my_red)
    ax.errorbar(N_arr, t_adjugate_arr[:, 0], yerr = t_adjugate_arr[:, 1], color = my_brown)

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.ylim(0.0001, 100)
    plt.xlim(100, 10**4)

    x_quad = [2 * 10**2, 10**3, 10**3, 2 * 10**2]
    y_quad = [2 * 10**-4, 2 * 10**-4, 2 * 10**-4 * (10**3 / (2 * 10**2))**2, 2 * 10**-4]
    y_quad2 = [2 * 10**-4, 2 * 10**-4, 2 * 10**-4 * (10**3 / (2 * 10**2)), 2 * 10**-4]
    ax.plot(x_quad, y_quad, 'k', alpha = 1)
    ax.plot(x_quad, y_quad2, 'k', alpha = 0.5)

    ax.text(2.5 * 10**3, 0.02, r"Primal", color=my_blue, fontsize=20)
    ax.text(2.5 * 10**3, 0.08, r"Adjoint", color=my_red, fontsize=20)
    ax.text(2.5 * 10**3, 30, r"Rogers", color=my_brown, fontsize=20)
    ax.text(1.1 * 10**3, 0.0008, r"$p = 1$", color="k", fontsize=20, alpha = 0.5)
    ax.text(1.1 * 10**3, 0.006, r"$p = 2$", color="k", fontsize=20)

    ax.yaxis.set_label_coords(-0.25, 0.5)
    ax.set_xlabel(r'$n$', fontsize=20, rotation=0)
    ax.set_ylabel(r'Wall time, (sec)', fontsize=20, rotation=0)

    plt.tight_layout()
    plt.savefig('../R1_journal/figures/scalability_adjugate.pdf')

    plt.show()
