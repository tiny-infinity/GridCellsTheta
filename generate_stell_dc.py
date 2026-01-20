import numpy as np
import copy
import json
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
        "sim_id": "PHPC_inhib_sweep_0_1",
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

base_input_params = generate_input_params()

range_inhib_dc = np.linspace(1e-3,4e-3,25)

all_param_sets = []

for i in range(len(range_inhib_dc)):
    param_set = copy.deepcopy(base_input_params)
    val = range_inhib_dc[i]
    param_set["global_inhib_dc"] = val
    param_set["sim_id"] = f"PHPC_inhib_sweep_0_{i+1}"
    all_param_sets.append(param_set)

parent_dir = "specs/phpc/"
new_specs_dir = parent_dir + "stell_inhib_sweeps_0/"

try:
    os.makedirs(new_specs_dir,exist_ok=True)
    print(f"Created directory at {new_specs_dir} or already exists")
except OSError as e:
    print(f'Error creating directory')

for i,param_set in enumerate(all_param_sets):
    file_name = new_specs_dir + f"PHPC-INHIB-SWEEP-0-{i+1}.py"
    with open(file_name,'w') as f:
        f.write("import numpy as np\n")
        f.write("import subprocess\n")
        f.write("import os\n")
        f.write("import time\n")
        f.write("import sys\n\n\n")
        f.write("def generate_input_params():\n")
        f.write(f"    sim_dur = float({param_set['sim_dur']})\n")
        f.write("    input_params = {\n")
        for key,val in param_set.items():
            if isinstance(val,str):
                f.write(f"        \"{key}\": \"{val}\",\n")
            else:
                f.write(f"        \"{key}\": {val},\n")
        f.write("    }\n\n")
        f.write("    return input_params\n")
    

