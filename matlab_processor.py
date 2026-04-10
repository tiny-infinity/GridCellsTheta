import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt, hilbert
import re
import matplotlib.pyplot as plt
from scipy.stats import linregress

class GridCellAnalyzer:
    def __init__(self, file_path, cell_idx, track_len=320):
        self.file_path = file_path
        self.track_len = track_len
        self.half_len = track_len / 2
        self.data = self._load_data(cell_idx)
        self._filter_theta()

    def _load_data(self, cell_idx):
        mat = sio.loadmat(self.file_path)
        pos_t = mat['post'].flatten()
        dur_sec = pos_t[-1]
        
        # Regex to find spike time keys (e.g., spks_t1c1)
        cell_keys = [k for k in mat.keys() if re.search(r'spks_t\dc\d', k, re.IGNORECASE)]
        selected_key = cell_keys[cell_idx]
        spikes = mat[selected_key].flatten()
        
        return {
            'pos_x': mat['posx'].flatten(),
            'pos_t': pos_t,
            'eeg': mat['EEG'].flatten(),
            'fs': float(mat['Fs'][0][0]),
            'spikes': spikes[spikes <= dur_sec],
            'cell_id': selected_key
        }

    def _filter_theta(self):
        # 6-10 Hz Bandpass filter for EEG theta 
        taps = firwin(501, [6, 10], pass_zero=False, fs=self.data['fs'], window='hamming')
        self.theta = filtfilt(taps, 1.0, self.data['eeg'])
        
        # Hilbert Transform for 0-360 degree phase assignment
        analytic_signal = hilbert(self.theta)
        phase = np.angle(analytic_signal) 
        self.phase_deg = np.rad2deg(phase) % 360 

    def analyze_direction(self, direction='out'):
        # Velocity calculation and filtering
        dx = np.diff(self.data['pos_x'])
        dt = np.diff(self.data['pos_t'])
        vel = np.append(dx/dt, (dx/dt)[-1])
        
        # Determine movement direction mask
        speed_mask = (vel > 10.0) if direction == 'out' else (vel < -10.0)
        
        # 1D Rate Map for field detection
        bin_size = 5
        num_bins = int(self.track_len / bin_size)
        bin_edges = np.linspace(-self.half_len, self.half_len, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        occ, _ = np.histogram(self.data['pos_x'][speed_mask], bins=bin_edges)
        occ = occ / 50.0 # Standard sampling rate conversion
        
        spike_vel = np.interp(self.data['spikes'], self.data['pos_t'], vel)
        dir_filter = (spike_vel > 10.0) if direction == 'out' else (spike_vel < -10.0)
        dir_spks = self.data['spikes'][dir_filter]
        spk_x = np.interp(dir_spks, self.data['pos_t'], self.data['pos_x'])
        
        spk_count, _ = np.histogram(spk_x, bins=bin_edges)
        rate_1d = np.divide(spk_count, occ, out=np.zeros_like(spk_count, dtype=float), where=occ!=0)
        
        return self._extract_fields(rate_1d, bin_centers, dir_spks, spk_x)

    def _extract_fields(self, rate_1d, centers, dir_spks, spk_x):
        max_rate = np.max(rate_1d)
        fields = []
        visited = np.zeros(len(rate_1d), dtype=bool)

        for i in range(len(rate_1d)-2):
            if visited[i] or rate_1d[i] < 0.1 * max_rate: continue
            
            # Find start and end where rate drops below 10% peak
            start, end = i, i
            while start > 0 and rate_1d[start-1] > 0.1 * max_rate: start -= 1
            while end < len(rate_1d)-1 and rate_1d[end+1] > 0.1 * max_rate: end += 1
            
            x_range = (centers[start], centers[end])
            visited[np.arange(start, end+1)] = True
            
            # Validation and Phase extraction
            spikes_in_mask = (spk_x >= x_range[0]) & (spk_x <= x_range[1])
            if np.sum(spikes_in_mask) < 40: continue 
            
            field_spk_times = dir_spks[spikes_in_mask]
            eeg_t = np.linspace(0, self.data['pos_t'][-1], len(self.phase_deg))
            field_spk_phase = np.interp(field_spk_times, eeg_t, self.phase_deg)
            
            fields.append({
                'x_abs': spk_x[spikes_in_mask],
                'phase': field_spk_phase,
                'bounds': x_range
            })
        return fields

def quantify_precession(field_data):
    # Quantify using absolute X to get slope in deg/cm
    x = field_data['x_abs']
    phases = field_data['phase']
    
    best_r2, best_slope, best_intercept, best_shift = -1, 0, 0, 0
    
    # Standard 360-degree rotation to find best linear fit for circular data
    for shift in range(0, 360, 2):
        shifted_phases = (phases + shift) % 360
        slope, intercept, r_val, _, _ = linregress(x, shifted_phases)
        if r_val**2 > best_r2:
            best_r2, best_slope, best_intercept, best_shift = r_val**2, slope, intercept, shift
            
    return {'slope': best_slope, 'r2': best_r2, 'shift': best_shift, 'intercept': best_intercept}

def plot_moser_style(fields, direction_label, cell_id):
    """
    Plots absolute position vs phase (720 deg) to match Moser Lab papers.
    """
    if not fields:
        print(f"No fields for {direction_label}")
        return

    plt.figure(figsize=(14, 6))
    
    for i, field in enumerate(fields):
        x = field['x_abs']
        p = field['phase']
        
        # Plot two cycles (0-720)
        plt.scatter(x, p, color='black', s=12, alpha=0.6)
        plt.scatter(x, p + 360, color='black', s=12, alpha=0.6)
        
        # Optional: Add regression line for each field
        res = quantify_precession(field)
        line_x = np.array([field['bounds'][0], field['bounds'][1]])
        # Note: Intercept needs careful handling with the shift, 
        # showing it here primarily for the scatter trend.
        
    plt.title(f"Phase Precession {direction_label.upper()} | {cell_id}")
    plt.xlabel("Position (cm)")
    plt.ylabel("Theta Phase (deg)")
    plt.ylim(0, 720)
    plt.yticks([0, 180, 360, 540, 720])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Execution ---
# Replace path with your actual filename
analyzer = GridCellAnalyzer("hafting_grid_cell_data/l2c5_0.mat", cell_idx=0)
in_fields = analyzer.analyze_direction('in')
out_fields = analyzer.analyze_direction('out')

plot_moser_style(in_fields, 'in', analyzer.data['cell_id'])
plot_moser_style(out_fields, 'out', analyzer.data['cell_id'])