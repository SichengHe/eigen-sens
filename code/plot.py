import numpy as np
from plot_utils import *
import matplotlib.pyplot as plt



def shiftAndScale(x, y, conn, x0_tar, y0_tar, chordLength_tar):

    x_final = np.zeros_like(x)
    y_final = np.zeros_like(y)

    # step 1: shit the coord to (0, 0)
    leadingEdge_ind = np.where(x == x.min())[0][0]

    deltaX = - x[leadingEdge_ind]
    deltaY = - y[leadingEdge_ind]
    print("leadingEdge_ind", leadingEdge_ind)
    print("y[leadingEdge_ind]", y[leadingEdge_ind])

    x_final[:] = x[:] + deltaX
    y_final[:] = y[:] + deltaY

    # step 2: scale the coord 
    leadingEdge_ind = np.where(x_final == x_final.min())[0][0]
    trailingEdge_ind = np.where(x_final == x_final.max())[0][0]

    x_trailing = x_final[trailingEdge_ind]
    y_trailing = y_final[trailingEdge_ind]
    x_leading = x_final[leadingEdge_ind]
    y_leading = y_final[leadingEdge_ind]

    print("deltaY, trailingEdge_ind, x_trailing, y_trailing, x_leading, y_leading", deltaY, trailingEdge_ind, x_trailing, y_trailing, x_leading, y_leading)

    chord = np.sqrt((x_trailing - x_leading)**2 + (y_trailing - y_leading)**2)
    scale = chordLength_tar / chord

    x_final *= scale
    y_final *= scale

    # step 3: shit to target coord
    x_final += x0_tar
    y_final += y0_tar

    return x_final, y_final

def plot_airfoil(ax, x, y, conn, color = 'k', linewidth = 4):

    x_tar = 0.0
    y_tar = 0.75
    chord_tar = 2.0

    x_final, y_final = shiftAndScale(x, y, conn, x_tar, y_tar, chord_tar)

    N = conn.shape[0]
    for i in range(N):
        ind1 = conn[i, 0] - 1
        ind2 = conn[i, 1] - 1

        ax.plot([x_final[ind1], x_final[ind2]], [y_final[ind1], y_final[ind2]], color = color, linewidth = linewidth)

def plot_cp(ax, x, y, cp, conn, color = 'k', linewidth = 4):

    # scale 
    x_tar = 0.0
    y_tar = 0.75 # not used
    chord_tar = 2.0

    x_final, y_final = shiftAndScale(x, y, conn, x_tar, y_tar, chord_tar)

    N = conn.shape[0]
    for i in range(N):
        ind1 = conn[i, 0] - 1
        ind2 = conn[i, 1] - 1

        ax.plot([x_final[ind1], x_final[ind2]], [cp[ind1], cp[ind2]], color = color, linewidth = linewidth)

    

# load data
baseline_dict = {}
optimized_dict = {}
timeinstances = ["1", "2", "3"]
slices = ["1", "2", "3"]
for i in range(len(timeinstances)):

    baseline_dict[timeinstances[i]] = {}
    optimized_dict[timeinstances[i]] = {}

    for j in range(len(slices)):

        baseline_dict[timeinstances[i]][slices[j]] = {}
        optimized_dict[timeinstances[i]][slices[j]] = {}

        baseline_dict[timeinstances[i]][slices[j]]["filename"] = "baseline_time" + timeinstances[i] + "_slice" + slices[j] + ".dat"
        optimized_dict[timeinstances[i]][slices[j]]["filename"] = "optimized_time" + timeinstances[i] + "_slice" + slices[j] + ".dat"

        data_local = np.loadtxt(baseline_dict[timeinstances[i]][slices[j]]["filename"])
        baseline_dict[timeinstances[i]][slices[j]]["x"] = data_local[:, 0]
        baseline_dict[timeinstances[i]][slices[j]]["z"] = data_local[:, 2]
        baseline_dict[timeinstances[i]][slices[j]]["cp"] = data_local[:, -3]

        data_local = np.loadtxt(optimized_dict[timeinstances[i]][slices[j]]["filename"])
        optimized_dict[timeinstances[i]][slices[j]]["x"] = data_local[:, 0]
        optimized_dict[timeinstances[i]][slices[j]]["z"] = data_local[:, 2]
        optimized_dict[timeinstances[i]][slices[j]]["cp"] = data_local[:, -3]



file_conns = ["topology1.dat", "topology2.dat", "topology3.dat"]
conns = [np.loadtxt(file_conn).astype('int') for file_conn in file_conns ]

# plot
fig, ((ax11, ax12, ax13), (ax21, ax22, ax23), (ax31, ax32, ax33)) = plt.subplots(3, 3, figsize=(12,12))
ax_list = [[ax11, ax12, ax13], [ax21, ax22, ax23], [ax31, ax32, ax33]]


N_timeinstances = len(timeinstances)
N_slices = len(slices)
for i in range(N_timeinstances):
    for j in range(N_slices):
        ax = ax_list[i][j]

        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(1.2, -1.2)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.get_xaxis().set_visible(False)

        ax.set_ylabel(r'$C_p$', fontsize=20, rotation=0)

        conn = conns[j]

        timeinstance_local = timeinstances[i]
        slice_local = slices[j]
        base_x = baseline_dict[timeinstance_local][slice_local]["x"]
        base_z = baseline_dict[timeinstance_local][slice_local]["z"]
        base_cp = baseline_dict[timeinstance_local][slice_local]["cp"]
        opt_x = optimized_dict[timeinstance_local][slice_local]["x"]
        opt_z = optimized_dict[timeinstance_local][slice_local]["z"]
        opt_cp = optimized_dict[timeinstance_local][slice_local]["cp"]
        plot_airfoil(ax, base_x, base_z, conn, color = my_blue, linewidth = 2)
        plot_cp(ax, base_x, base_z, base_cp, conn, color = my_blue)
        plot_airfoil(ax, opt_x, opt_z, conn, color = my_green, linewidth = 2)
        plot_cp(ax, opt_x, opt_z, opt_cp, conn, color = my_green)


plt.savefig('../../cp_wing.pdf',bbox_inches='tight')

plt.show()