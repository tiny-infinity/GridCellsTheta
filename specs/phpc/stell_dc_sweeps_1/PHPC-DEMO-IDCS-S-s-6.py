import numpy as np
import subprocess
import os
import time
import sys


def generate_input_params():
    sim_dur = float(35000.0)
    input_params = {
        "sim_dur": 35000.0,
        "N_intrnrn": 193,
        "sim_id": "PHPC_stell_sweep_0_6",
        "Amp_i_theta": 0.0001035,
        "omega_i_theta": 0.01,
        "conn_id": "1d_phpc",
        "instr_id": "1d_phpc",
        "phi_i_theta": 0,
        "n_cpus": 4,
        "ext_Amp_i_theta": 0.0001035,
        "ext_is_peak": 2,
        "intrnrn_init_noise": [100, 0, 0.5],
        "stell_init_noise": [100, 0, 0.5],
        "intrnrn_noise": [150, 0, 0.002],
        "stell_noise": [150, 0, 0.001],
        "intrnrn_dc_amp": 0.0015,
        "stell_const_dc": [0.0018357142857142851, -0.00275],
        "global_inhib_dc": 0.002224,
        "stell_theta_Amp": 0,
        "stell_theta_omega": 0.012,
        "init_noise_seed": 30079,
        "noise_seed": 21404,
        "record_handle_intrnrn": {'intrnrn_v': {'state': True, 'cells_to_record': 'all'}, 'intrnrn_theta_i': {'state': True, 'cells_to_record': 'all'}},
        "record_handle_stell": {'stell_v': {'state': True, 'cells_to_record': 'all'}, 'stell_ina': {'state': False, 'cells_to_record': 'all'}, 'stell_theta_i': {'state': False, 'cells_to_record': 'all'}},
    }

    return input_params
