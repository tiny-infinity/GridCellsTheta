"""
Adjacency matrix for base 1D network.
The network consists of three rings 2 stellate and 1 inhibitory connected 
through mexican hat-type connection.
-----------
|s->s|s->i|
|i->s|i->i|
-----------
"""
import numpy as np
import h5py
import conn_utils


args = conn_utils.file_args().parse_args()
conn_utils.add_project_to_sys_path() #to access parent directories
params = conn_utils.find_params(args)


conn_id = params['conn_id']
sim_id = params['sim_id']
sim_num = params['sim_num']
n_intrnrn = N_per_sheet = params['N_intrnrn']-1
n_stell = params['N_stell']
si_peak = params['si_peak']
ii_peak = params['ii_peak']
is_peak = params['is_peak']
si_mean = params['si_mean']*((2*np.pi)/N_per_sheet)
si_stdev = params['si_stdev']
is_stdev = params['is_stdev']
is_mean = params['is_mean']*((2*np.pi)/N_per_sheet)
ii_stdev = params['ii_stdev']
ii_mean = params['ii_mean']*((2*np.pi)/N_per_sheet)
threshold = 1e-7



#--------------------------------------------------------------------------------#

cnnct_ss = np.zeros((n_stell, n_stell))

#--------------------------------------------------------------------------------#
cnnct_si = np.zeros((n_stell, n_intrnrn))

for i in range(N_per_sheet):
    for j in range(N_per_sheet):
        theta_i,theta_j = (2*np.pi/N_per_sheet)*i,(2*np.pi/N_per_sheet)*j
        delta_theta=conn_utils.signed_arc_length(theta_i,theta_j)
        cnnct_si[i, j] = conn_utils.gaussian(delta_theta, si_peak, si_mean, si_stdev)

cnnct_si[cnnct_si < threshold] = 0
temp_mat = np.zeros((N_per_sheet, N_per_sheet))
for i in range(N_per_sheet):
    for j in range(n_intrnrn):
        theta_i,theta_j = (2*np.pi/N_per_sheet)*i,(2*np.pi/N_per_sheet)*j
        delta_theta=conn_utils.signed_arc_length(theta_i,theta_j)
        temp_mat[i, j] = conn_utils.gaussian(delta_theta, si_peak, -si_mean, si_stdev)
cnnct_si[N_per_sheet:, :] = temp_mat
cnnct_si[np.abs(cnnct_si) < threshold] = 0


#---------------------------------------------------------------------------------#
cnnct_is = np.zeros((n_intrnrn, n_stell))

for i in range(n_intrnrn):
    for j in range(N_per_sheet):
        theta_i,theta_j = (2*np.pi/N_per_sheet)*i,(2*np.pi/N_per_sheet)*j
        delta_theta=conn_utils.unsigned_arc_length(theta_i,theta_j)
        cnnct_is[i, j] = conn_utils.gaussian(delta_theta, is_peak, is_mean, is_stdev)

cnnct_is[:, N_per_sheet:] = cnnct_is[:, :N_per_sheet]
cnnct_is[np.abs(cnnct_is) < threshold] = 0

#----------------------------------------------------------------------------------#

cnnct_ii = np.zeros((n_intrnrn, n_intrnrn))

for i in range(n_intrnrn):
    for j in range(n_intrnrn):
        theta_i,theta_j = (2*np.pi/N_per_sheet)*i,(2*np.pi/N_per_sheet)*j
        delta_theta=conn_utils.unsigned_arc_length(theta_i,theta_j)
        cnnct_ii[i, j] = conn_utils.gaussian(delta_theta, ii_peak, ii_mean, ii_stdev)

cnnct_ii[np.abs(cnnct_ii) < threshold] = 0


#----------------------------------------------------------------------------------#

adj_matrix = np.hstack(
    (np.vstack((cnnct_ss, cnnct_is)), np.vstack((cnnct_si, cnnct_ii))))


temp = np.vstack((np.zeros(n_stell)[:,None],np.zeros(n_intrnrn)[:,None]))
adj_matrix = np.hstack((adj_matrix,temp))
temp = np.hstack((np.full(N_per_sheet,params["ext_is_peak"]),np.zeros(N_per_sheet),np.zeros(n_intrnrn+1)))
adj_matrix = np.vstack((adj_matrix,temp))

 
if params["save_conn_matrix"]:
    with h5py.File(f"network_configs/connections/saved_matrices/matrix_{conn_id}_{params['matrix_id']}.hdf5", "w") as f:
        z=f.create_dataset(str('matrix'),data=adj_matrix,compression='gzip')

with h5py.File(f"cache/matrix_{conn_id}_{sim_id}_{sim_num}.hdf5", "w") as f:
    z=f.create_dataset(str('matrix'),data=adj_matrix,compression='gzip') 

