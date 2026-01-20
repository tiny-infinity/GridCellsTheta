"""
Trajectory1D class is used to generate different types of input to the network.
    - Constant velocity
    - Input velocity: variable velocity with allothetic input
    - Constant velocity for predictive coding
    - Pulse input 
        for raster plot in figure 1 of the paper
"""

import numpy as np
import h5py as hdf
import pickle
from scipy.interpolate import BSpline
import sim_utils as s_utils
from neuron import h

class Trajectory1D:
    def __init__(self, params, save_mem=True):
        # self.max_vel = 35 * 1e-3  # cm/ms
        self.params = params
        self.sim_dur = params["sim_dur"]
        self.dt = params["dt"]
        self.t = np.arange(0, self.sim_dur + self.dt, self.dt)
        self.N_per_sheet = self.params["N_per_sheet"]
        self.save_mem = save_mem
        self.active_cells = []
        if self.params["vel_type"] == "const":
            self.const_vel()
        elif self.params["vel_type"] == "input":
            self.input_velocity()
        elif self.params["vel_type"] == "ACVT-1DAC":
            self.figure_1_pulse_input()
        elif self.params["vel_type"] == "PRED-IHD":
            self.predictive_inputs()



    def vel_to_dc(self, vel_input, min_dc, max_dc):
        return ((max_dc - min_dc) / self.max_vel) * vel_input + min_dc

    def vel_to_dc_fit(self, input_vel):
        #spline
        spline_params = s_utils.json_read("input_data/vi_transform/spline_params.json")
        dc_out=BSpline(*list(spline_params.values()))(input_vel)
        other_ring_mask = np.isnan(input_vel)
        dc_out[other_ring_mask] = self.params["vel_integ_or"]

        return dc_out

    def decompose_vel(self):
        self.right_vel = self.vel_input.copy()
        self.left_vel = self.vel_input.copy()
        self.right_vel[self.right_vel < 0] = np.nan
        self.left_vel[self.left_vel > 0] = np.nan

    def const_vel(self):
        self.intrnrn_dc=np.full_like(self.t, self.params["intrnrn_dc_amp"])
        self.right_const_dc = self.params["stell_const_dc"][0]
        self.left_const_dc = self.params["stell_const_dc"][1]
        self.left_dc = np.full_like(self.t, self.left_const_dc)
        self.right_dc = np.full_like(self.t, self.right_const_dc)


    def input_velocity(self):

        with hdf.File("input_data/trajectories/traj_{}.hdf5".format(self.params["traj_id"]), "r") as file:
            self.vel_input = np.array(file["vel_rinb"][:])
            self.pos_input = np.array(file["pos_rinb"][:])
            self.init_allothetic_dur = float(file.attrs["allothetic_dur"])
        self.decompose_vel()
        self.right_dc = self.vel_to_dc_fit(self.right_vel)
        self.left_dc = self.vel_to_dc_fit(-1 * self.left_vel)
        self.intrnrn_dc = np.full_like(self.t, self.params["intrnrn_dc_amp"])
        if self.params["init_allothetic_input"]:
            self.allothetic_input()

        if self.save_mem:
            del self.vel_input, self.right_vel, self.left_vel, self.pos_input



    def allothetic_input(self):
        # get t_idx at which allothetic input ends (in dt units)
        self.init_idx = int((self.init_allothetic_dur) / 0.025)

        #get the input position when allothetic input ends
        self.init_position = self.pos_input[
            self.init_idx
        ]

        self.allothetic_left_dc = self.left_dc.copy()
        self.allothetic_right_dc = self.right_dc.copy()
        
        #set DC input for the allothetic dur (gradual increase)
        self.allothetic_left_dc[: self.init_idx] = np.linspace(1e-2,self.left_dc[self.init_idx],self.allothetic_left_dc[: self.init_idx].shape[0])
        self.allothetic_right_dc[: self.init_idx] = np.linspace(1e-2,self.right_dc[self.init_idx],self.allothetic_right_dc[: self.init_idx].shape[0])
        
        #set DC inputs for all the other cells that are not part of the allothetic input
        self.left_dc[: self.init_idx+int(0/0.025)] =self.params["allothetic_stell_dc"]
        self.right_dc[: self.init_idx+int(0/0.025)]=self.params["allothetic_stell_dc"]
        
        #Find cells that should be active during allothetic input
        phi = np.round(
            (self.init_position*self.params["n_phases"]) / (self.params["lambda0"])
        ).astype("int")
        t_active_grids = ((np.arange(int(self.N_per_sheet /self.params["n_phases"])) * \
        self.params["n_phases"])[:,None] +np.arange(-self.params["allothetic_nrn_n"],self.params["allothetic_nrn_n"]+1)+phi)%self.N_per_sheet
        t_active_grids=t_active_grids.ravel()
        self.active_cells= t_active_grids + self.params["N_stell"]

        #set DC input for active interneurons for allothetic dur
        self.allothetic_intrnrn_dc = self.intrnrn_dc.copy()
        self.allothetic_intrnrn_dc[: self.init_idx] =1.5e-3
        self.intrnrn_dc[: self.init_idx] =-1e-3
        
        #convert to neuron vectors
        self.ext_amp_right_allo = h.Vector(self.allothetic_right_dc)
        self.ext_amp_left_allo = h.Vector(self.allothetic_left_dc)
        self.ext_amp_intnrn_allo = h.Vector(self.allothetic_intrnrn_dc)

    def figure_1_pulse_input(self):
        "Used to generate a pulse input that is used for raster plot in figure one of the paper"
        # define step inputs
        x = [0, 2000, 4000, 6000]
        left_ring = [-1.5e-3, -3e-3, 0]
        right_ring = [-1.5e-3, 0, -3e-3]
        self.left_dc = self.create_piecewise(x, left_ring)
        self.right_dc = self.create_piecewise(x, right_ring)
        self.intrnrn_dc=np.full_like(self.t, self.params["intrnrn_dc_amp"])

    def create_piecewise(self, x, y):
        t_ = np.arange(0, self.sim_dur + self.dt, self.dt)
        l = []
        for i in range(len(x) - 1):
            l.append(np.logical_and(t_ >= x[i], t_ <= x[i + 1]))

        return np.piecewise(t_, l, y)
    
    def predictive_inputs(self):
        """Pulse inputs for predictive coding (no initial input)"""
        step_curr_t = np.arange(0, self.sim_dur + self.dt, self.dt)
        # define step inputs
        x = [0, self.params["extra_params"]["dir_change_t"], self.params["sim_dur"]]
        left_ring = [-3e-2, self.params["extra_params"]["stell_dc"]]
        right_ring = [self.params["extra_params"]["stell_dc"], -3e-2]
        self.left_dc = self.create_piecewise(x, left_ring)
        self.right_dc = self.create_piecewise(x, right_ring)
        self.intrnrn_dc=np.full_like(self.t, self.params["intrnrn_dc_amp"])