import subprocess
import os
import time
import sys
import numpy as np



def generate_input_params():
    sim_dur = float(35000)
    input_params = {
        "sim_dur": sim_dur,
        "N_intrnrn": 192+1,
        "sim_id": "PHPC-DEMO-IDCS-S-s-1a",
        "Amp_i_theta":1.035e-4,
        "omega_i_theta": 0.01,
        "conn_id":"1d_phpc",
        "instr_id":"1d_phpc",
        "phi_i_theta":0,
        "n_cpus":4,
        "ext_Amp_i_theta":1.035e-4,
        "ext_is_peak":2,
        "intrnrn_init_noise":[100,0,0.5],
        "stell_init_noise":[100,0,0.5],
        "intrnrn_noise":[150,0,2e-3],
        "stell_noise":[150,0,1e-3],
        "intrnrn_dc_amp":1.5e-3,
        "stell_const_dc":[4e-3,-2.75e-3],
        "global_inhib_dc":2.224e-3,
        "stell_theta_Amp":0,
        "stell_theta_omega":0.012,
        "init_noise_seed":30079,
        "noise_seed":21404,
        "record_handle_intrnrn":{"intrnrn_v": {"state": True,"cells_to_record":"all"},
                                 "intrnrn_theta_i": {"state": True,"cells_to_record":"all"}},
        "record_handle_stell":{"stell_v": {"state": True,"cells_to_record":"all"},
                               "stell_ina": {"state": False,"cells_to_record":"all"},
                               "stell_theta_i": {"state": False,"cells_to_record":"all"}}
    }

    return input_params