import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt, hilbert
import re
import matplotlib.pyplot as plt
# Custom palette from your poster
poster_colors = ['#2D5A82', '#A6192E', '#4A90E2', '#9013FE', '#417505']

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=poster_colors)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['axes.edgecolor'] = '#2D5A82'

class HighDensityGridAnalyzer:
    def __init__(self, file_path, cell_idx, track_len=320):
        self.file_path = file_path
        self.track_len = track_len
        self.half_len = track_len / 2
        self.data = self._load_data(cell_idx)
        self._filter_theta()

    def _load_data(self, cell_idx):
        mat = sio.loadmat(self.file_path)
        pos_t = mat['post'].flatten()
        
        # Extract cell keys
        cell_keys = [k for k in mat.keys() if re.search(r'spks_t\dc\d', k, re.IGNORECASE)]
        selected_key = cell_keys[cell_idx]
        spikes = mat[selected_key].flatten()
        
        return {
            'pos_x': mat['posx'].flatten(),
            'pos_t': pos_t,
            'eeg': mat['EEG'].flatten(),
            'fs': float(mat['Fs'][0][0]),
            'spikes': spikes[spikes <= pos_t[-1]],
            'cell_id': selected_key
        }

    def _filter_theta(self):
        # 6-10 Hz Bandpass
        taps = firwin(501, [6, 10], pass_zero=False, fs=self.data['fs'], window='hamming')
        theta = filtfilt(taps, 1.0, self.data['eeg'])
        self.theta_sig = theta
        # Hilbert for Phase
        analytic = hilbert(theta)
        self.phase_deg = np.rad2deg(np.angle(analytic)) % 360 

    def plot_theta(self,cell_idx,start,end):
        theta_sig = self._load_data(cell_idx)['eeg']
        theta_sig = self.theta_sig
        time_dur = self._load_data(cell_idx)['pos_t'][-1]
        time_arr = np.linspace(0,time_dur,len(theta_sig))
        plt.plot(time_arr,theta_sig)
        plt.xlim(start,end)
        plt.show()


    def get_directional_data(self, direction='out'):
        """
        Extracts ALL spikes for a direction without strict speed/field filtering
        to match the high-density supplementary figures.
        """
        # 1. Calculate Velocity
        dx = np.diff(self.data['pos_x'])
        dt = np.diff(self.data['pos_t'])
        vel = np.append(dx/dt, (dx/dt)[-1])
        
        # 2. Define Directional Mask (Low speed threshold to keep more spikes)
        if direction == 'out':
            dir_mask = vel > 2.0  # Just enough to ensure it's moving 'out'
            linear_pos = self.data['pos_x'] + self.half_len
        else:
            dir_mask = vel < -2.0 # Just enough to ensure it's moving 'in'
            linear_pos = self.half_len - self.data['pos_x']

        # 3. Map Spikes to Linearized Position and Phase
        # Interpolate velocity to spike times to filter spikes by direction
        spike_vel = np.interp(self.data['spikes'], self.data['pos_t'], vel)
        if direction == 'out':
            valid_spikes = self.data['spikes'][spike_vel > 0]
        else:
            valid_spikes = self.data['spikes'][spike_vel < 0]

        # Get position and phase for these spikes
        spk_x = np.interp(valid_spikes, self.data['pos_t'], linear_pos)
        
        # We need to map spike times to the EEG timebase for phase
        eeg_t = np.linspace(0, self.data['pos_t'][-1], len(self.phase_deg))
        spk_phase = np.interp(valid_spikes, eeg_t, self.phase_deg)
        
        return spk_x, spk_phase

def plot_full_track_precession(spk_x, spk_phase, direction, cell_id):
    plt.figure(figsize=(12, 6))
    
    # Plot two cycles (0-720) to show the 'slopes' clearly
    # Using small 's' and low 'alpha' to handle high spike density
    plt.scatter(spk_x, spk_phase, color='C0', s=3, alpha=1)
    plt.scatter(spk_x, spk_phase + 360, color='C0', s=3, alpha=1)
    
    plt.title(f"Session ID 13120410 | Cell ID : {cell_id} ({direction.upper()})")
    plt.xlabel("Linearized Distance from Start (cm)")
    plt.ylabel("Theta Phase (deg)")
    
    plt.xlim(0, 320)
    plt.ylim(0, 720)
    plt.yticks([0, 180, 360, 540, 720])
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('final_figs/HAFTING_t5c1.png',dpi=600)
    plt.show()

# --- Execution ---
analyzer = HighDensityGridAnalyzer("hafting_grid_cell_data/l2c5_0.mat", cell_idx=0)
analyzer.plot_theta(0,10,20)
# Process 'Out' direction
x_out, p_out = analyzer.get_directional_data('out')
plot_full_track_precession(x_out, p_out, 'out', analyzer.data['cell_id'])



