import numpy as np
"""
"LONG SIMULATION TIME ~ 11 hours with 10 nodes of 32 cores each. 
It consists of a grid search with 1,200 simulations, each having 4 trials. 
Simulation was run on a cluster."
"""
def generate_mult_input_params()-> dict:
    dc_range_arr = np.concatenate((np.linspace(-2.7e-3,0,1000,endpoint=False),
                                       np.linspace(0,2e-2,200)))
    n_trials = 4
    sim_num = 0
    sim_dur = float(60000)
    mult_input_params={}
    for dc in dc_range_arr:
        for tr in range(n_trials):
            input_params = {
                "sim_num":sim_num,
                "sim_dur": sim_dur,
                "sim_id": "VALD-PI-VICR-S-m-1a",
                "intrnrn_init_noise":[100,0,0.5],
                "stell_init_noise":[100,0,0.5],
                "stell_const_dc":[dc,-2.7e-3],
                "n_phases":64,
                "si_peak":2.43,
                "is_stdev":0.136,
                "ii_stdev":0.136,
                "init_noise_seed":50,
                "noise_seed":100,
                "n_nodes":10,
                "data_root":"data/",
                }
            mult_input_params[str(sim_num)] = input_params #Important.
            sim_num+=1
    return mult_input_params