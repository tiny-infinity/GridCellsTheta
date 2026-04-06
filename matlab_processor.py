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
        # 6-10 Hz Bandpass filter for local EEG theta 
        taps = firwin(501, [6, 10], pass_zero=False, fs=self.data['fs'], window='hamming')
        self.theta = filtfilt(taps, 1.0, self.data['eeg'])
        
        # Hilbert Transform for 0-360 degree phase assignment
        analytic_signal = hilbert(self.theta)
        phase = np.angle(analytic_signal) 
        self.phase_deg = np.rad2deg(phase) % 360 

    def analyze_direction(self, direction='out'):
        # Calculate Velocity and filter for speed > 10 cm/s 
        dx = np.diff(self.data['pos_x'])
        dt = np.diff(self.data['pos_t'])
        vel = np.append(dx/dt, (dx/dt)[-1])
        
        speed_mask = (vel > 10.0) if direction == 'out' else (vel < -10.0)
        
        # 1D Rate Map (5cm bins, No Smoothing) 
        bin_size = 5
        num_bins = int(self.track_len / bin_size)
        bin_edges = np.linspace(-self.half_len, self.half_len, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        occ, _ = np.histogram(self.data['pos_x'][speed_mask], bins=bin_edges)
        occ = occ / 50.0 
        
        spike_vel = np.interp(self.data['spikes'], self.data['pos_t'], vel)
        dir_filter = (spike_vel > 10.0) if direction == 'out' else (spike_vel < -10.0)
        dir_spks = self.data['spikes'][dir_filter]
        spk_x = np.interp(dir_spks, self.data['pos_t'], self.data['pos_x'])
        
        spk_count, _ = np.histogram(spk_x, bins=bin_edges)
        rate_1d = np.divide(spk_count, occ, out=np.zeros_like(spk_count, dtype=float), where=occ!=0)
        
        # Find and Validate Fields based on 10% peak/1% drop rules [cite: 710, 840]
        fields = self._extract_fields(rate_1d, bin_centers, dir_spks, spk_x)
    
        print(f"Direction: {direction.upper()} | Valid {direction}-fields found: {len(fields)}")
        
        return fields

    def _extract_fields(self, rate_1d, centers, dir_spks, spk_x):
        max_rate = np.max(rate_1d)
        fields = []
        visited = np.zeros(len(rate_1d), dtype=bool)
        distal_limit = self.half_len * 0.95 

        for i in range(len(rate_1d)-2):
            if visited[i] or not np.all(rate_1d[i:i+3] > 0.1 * max_rate): continue
            
            start, end = i, i+2
            while start > 0 and not (rate_1d[start-1] < 0.01*max_rate or rate_1d[start-1] > rate_1d[start]): start -= 1
            while end < len(rate_1d)-1 and not (rate_1d[end+1] < 0.01*max_rate or rate_1d[end+1] > rate_1d[end]): end += 1
            
            x_range = (centers[start], centers[end])
            visited[np.arange(start, end+1)] = True
            
            # Validation: 50 spike min and 5% distal exclusion 
            spikes_in_mask = (spk_x >= x_range[0]) & (spk_x <= x_range[1])
            if np.sum(spikes_in_mask) < 50 or abs(x_range[0]) > distal_limit or abs(x_range[1]) > distal_limit:
                continue 
            
            # Phase assignment for valid spikes [cite: 1443]
            field_spk_times = dir_spks[spikes_in_mask]
            eeg_t = np.linspace(0, self.data['pos_t'][-1], len(self.phase_deg))
            field_spk_phase = np.interp(field_spk_times, eeg_t, self.phase_deg)
            
            fields.append({
                'x': spk_x[spikes_in_mask],
                'phase': field_spk_phase,
                'normalized_x': (spk_x[spikes_in_mask] - x_range[0]) / (x_range[1] - x_range[0]) * 100
            })
        return fields
    
def quantify_precession_with_details(field_data):
    x = field_data['normalized_x']
    phases = field_data['phase']
    
    best_r2, best_slope, best_r, best_intercept, best_shift = -1, 0, 0, 0, 0
    
    for shift in range(360):
        shifted_phases = (phases + shift) % 360
        slope, intercept, r_val, _, _ = linregress(x, shifted_phases)
        r2 = r_val**2
        
        if r2 > best_r2:
            best_r2, best_slope, best_r, best_intercept, best_shift = r2, slope, r_val, intercept, shift
            
    return {
        'slope': best_slope,
        'intercept': best_intercept,
        'r2': best_r2,
        'correlation': best_r,
        'best_shift': best_shift
    }

def plot_fitted_precession(field_data, fit_results, direction_label):
    """
    Plots the normalized position vs. shifted phase with the regression line.
    """
    x = field_data['normalized_x']
    # Use the best shift found during quantification to align the spikes
    best_shift = fit_results['best_shift']
    shifted_phases = (field_data['phase'] + best_shift) % 720
    
    plt.figure(figsize=(8, 5))
    
    # Plot the spikes at the optimal rotation
    plt.scatter(x, shifted_phases, color='black', alpha=0.5, s=20, label='Spikes (Optimal Rotation)')
    
    # Plot the regression line
    # Generate x-values for the line (0 to 100%)
    line_x = np.array([0, 100])
    line_y = fit_results['slope'] * line_x + fit_results['intercept']
    
    plt.plot(line_x, line_y, color='red', linewidth=2, 
             label=f"Fit (R²={fit_results['r2']:.3f}, r={fit_results['correlation']:.2f})")
    
    plt.title(f"Quantified Phase Precession ({direction_label.upper()})")
    plt.xlabel("Position in Field (%)")
    plt.ylabel("Shifted Theta Phase (deg)")
    plt.ylim(0, 720)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_all_fields(fields, direction_label, cell_id):
    if not fields:
        print(f"No valid {direction_label} fields found for {cell_id}.")
        return

    plt.figure(figsize=(10, 6))

    colors = plt.cm.get_cmap('tab10', len(fields))
    
    for i, field in enumerate(fields):
        label = f"Field {i+1}"
        # Plot two cycles (0-720 degrees) 
        plt.scatter(field['normalized_x'], field['phase'], 
                    color=colors(i), alpha=0.4, s=15, label=label)
        plt.scatter(field['normalized_x'], field['phase'] + 360, 
                    color=colors(i), alpha=0.4, s=15)

    plt.title(f"Stacked Phase Precession ({direction_label.upper()}): {cell_id}")
    plt.xlabel("Position in Field (%)")
    plt.ylabel("Theta Phase (deg)")
    plt.ylim(0, 720) # Standard 0-720 degree view
    plt.xlim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


analyzer = GridCellAnalyzer("hafting_grid_cell_data/l2c3_3.mat", cell_idx=1, track_len=320)
in_fields = analyzer.analyze_direction('in')
out_fields = analyzer.analyze_direction('out')

plot_all_fields(in_fields, 'in', analyzer.data['cell_id'])
plot_all_fields(out_fields, 'out', analyzer.data['cell_id'])

if in_fields:
    for i, field in enumerate(in_fields):
        results = quantify_precession_with_details(field)
        print(f"Field {i+1} Slope: {results['slope']:.3f} deg/%")
        plot_fitted_precession(field, results, f"In-Field {i+1}")