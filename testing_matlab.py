import numpy as np
import scipy.io
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt


data = scipy.io.loadmat('hafting_grid_cell_data/l2c1_1.mat')
print(data)
eeg = data['EEG'].flatten()
cell_ts = data['spks_t5c1'].flatten() 
post = data['post'].flatten()

fs_eeg = 250 
duration = 600 
t_eeg = np.linspace(0, duration, len(eeg))

numtaps = 501 
taps = firwin(numtaps, [6, 10], pass_zero=False, fs=fs_eeg, window='hamming')

theta_filtered = filtfilt(taps, 1.0, eeg)

start_time = 346
end_time = 348

window_mask = (t_eeg >= start_time) & (t_eeg <= end_time)
spike_mask = (cell_ts >= start_time) & (cell_ts <= end_time)

plt.figure(figsize=(15, 6))
plt.plot(t_eeg[window_mask], eeg[window_mask], color='lightgray', label='Raw EEG', alpha=0.7)
plt.plot(t_eeg[window_mask], theta_filtered[window_mask], color='royalblue', linewidth=2, label='Filtered Theta (6-10 Hz)')


spike_height = np.max(theta_filtered[window_mask])
plt.vlines(cell_ts, -spike_height, spike_height, color='crimson', linewidth=2, label='Spike Events')
plt.title(f"EEG Theta Phase and Spike Alignment ({start_time}-{end_time}s)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)") 
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Total spikes in window: {np.sum(spike_mask)}")