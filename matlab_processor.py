import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt, hilbert
import re
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Custom palette from your poster
poster_colors = ['#2D5A82', '#A6192E', '#4A90E2', '#9013FE', '#417505']

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=poster_colors)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['axes.edgecolor'] = '#2D5A82'

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
        
        return self._extract_fields(rate_1d, bin_centers, dir_spks, spk_x,direction)

    def _extract_fields(self, rate_1d, centers, dir_spks, spk_x, direction):
        max_rate = np.max(rate_1d)
        fields = []
        visited = np.zeros(len(rate_1d), dtype=bool)

        for i in range(len(rate_1d)-2):
            if visited[i] or rate_1d[i] < 0.1 * max_rate: continue
            
            start, end = i, i
            while start > 0 and rate_1d[start-1] > 0.1 * max_rate: start -= 1
            while end < len(rate_1d)-1 and rate_1d[end+1] > 0.1 * max_rate: end += 1
            
            x_range = (centers[start], centers[end])
            visited[np.arange(start, end+1)] = True
            
            spikes_in_mask = (spk_x >= x_range[0]) & (spk_x <= x_range[1])
            if np.sum(spikes_in_mask) < 40: continue 
            
            field_spk_times = dir_spks[spikes_in_mask]
            eeg_t = np.linspace(0, self.data['pos_t'][-1], len(self.phase_deg))
            field_spk_phase = np.interp(field_spk_times, eeg_t, self.phase_deg)
            
            # --- DIRECTIONALITY FIX ---
            # If moving 'in', negate the x-coordinates so 'start' of travel 
            # is mathematically smaller than the 'end' of travel.
            x_for_regression = spk_x[spikes_in_mask]
            if direction == 'in':
                x_for_regression = -x_for_regression
            
            fields.append({
                'x_abs': spk_x[spikes_in_mask],       # Keep absolute for plotting
                'x_reg': x_for_regression,           # Use this for slope calculation
                'phase': field_spk_phase,
                'bounds': x_range,
                'direction': direction
            })
        return fields

def quantify_precession(field_data):
    # Use x_reg to ensure slope sign is direction-independent
    x = field_data['x_reg']
    phases = field_data['phase']
    
    best_r2, best_slope, best_intercept, best_shift = -1, 0, 0, 0
    
    for shift in range(0, 360, 2):
        shifted_phases = (phases + shift) % 360
        slope, intercept, r_val, _, _ = linregress(x, shifted_phases)
        if r_val**2 > best_r2:
            best_r2, best_slope, best_intercept, best_shift = r_val**2, slope, intercept, shift
            
    return {'slope': best_slope, 'r2': best_r2, 'shift': best_shift, 'intercept': best_intercept}

def plot_moser_style(fields, direction_label, cell_id):
    if not fields:
        print(f"No fields for {direction_label}")
        return

    # Create figure - if multiple fields, you might want subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, field in enumerate(fields):
        x = field['x_abs']
        p = field['phase']
        
        # 1. Get regression stats for this field
        stats = quantify_precession(field)
        slope = stats['slope']
        r2 = stats['r2']
        shift = stats['shift']
        intercept = stats['intercept']
        
        # 2. Plot the double-cycle points (0-720)
        # We use a specific color from your poster palette
        ax.scatter(x, p, color='#2D5A82', s=15, alpha=0.6, label='Spikes' if i==0 else "")
        ax.scatter(x, p + 360, color='#2D5A82', s=15, alpha=0.3)
        
        # 3. Calculate the regression line
        # The regression was: (phase + shift) % 360 = slope * x + intercept
        # To plot it, we create an x-range across the field bounds
        x_range = np.linspace(np.min(x), np.max(x), 100)
        y_line_shifted = slope * x_range + intercept
        
        # To align the line with the 'p' or 'p+360' scatter points:
        # We adjust the line back by the shift used in the regression
        y_line_raw = (y_line_shifted - shift)
        
        # Determine which cycle the line fits best visually
        # (This handles the 0-720 wrap-around for the line display)
        mean_p = np.mean(p)
        if np.abs(np.mean(y_line_raw) - mean_p) > np.abs(np.mean(y_line_raw + 360) - mean_p):
            y_line_raw += 360
            
        ax.plot(x_range, y_line_raw, color='#A6192E', linewidth=2, 
                label=f"Field {i+1} Fit: {slope:.2f} deg/cm")

    # Formatting to match Hafting et al. (2008)
    plt.title(f"Cell: {cell_id} | Dir: {direction_label.upper()}\n"
              f"Mean Slope: {slope:.2f} $^\circ$/cm | $R^2$: {r2:.3f}")
    
    plt.xlabel("Position (cm)")
    plt.ylabel("Theta Phase ($^\circ$)")
    plt.ylim(0, 720)
    plt.yticks([0, 180, 360, 540, 720])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.show()



# --- Execution ---
# Replace path with your actual filename
analyzer = GridCellAnalyzer("hafting_grid_cell_data/l2c5_0.mat", cell_idx=0)
in_fields = analyzer.analyze_direction('in')
out_fields = analyzer.analyze_direction('out')

# Update your call to ensure the regression result is accessible

plot_moser_style(in_fields, 'in', analyzer.data['cell_id'])
plot_moser_style(out_fields, 'out', analyzer.data['cell_id'])
