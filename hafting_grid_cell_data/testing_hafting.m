% 1. Interpolate position to find where the rat was during each spike
spikeX = interp1(post, posx, cellTS);
spikeY = interp1(post, posy, cellTS);

% 2. Plot the trajectory (grey) and spikes (red dots)
figure;
plot(posx, posy, 'Color', [0.8 0.8 0.8]); % Trajectory in light grey [cite: 56]
hold on;
plot(spikeX, spikeY, 'r.', 'MarkerSize', 10); % Spikes in red
xlabel('X Position (cm)');
ylabel('Y Position (cm)');
title('Spatial Firing Map');
axis equal;

% Calculate instantaneous speed (cm/s)
dx = diff(posx);
dy = diff(posy);
dt = diff(post);
speed = sqrt(dx.^2 + dy.^2) ./ dt;

% Plot speed over time
figure;
plot(post(1:end-1), speed);
hold on;
yline(10, 'r--', 'Threshold (10 cm/s)'); % 
xlabel('Time (s)');
ylabel('Speed (cm/s)');
title('Running Speed Profile');

% Create a time axis for the EEG (assuming 250Hz based on 150,000 samples / 600s)
eeg_sampling_rate = 250; 
t_eeg = (0:length(EEG)-1) / eeg_sampling_rate;

% Plot a 2-second window to see individual theta cycles
window = [100 115]; % Seconds 100 to 102
idx = t_eeg >= window(1) & t_eeg <= window(2);
spike_idx = cellTS >= window(1) & cellTS <= window(2);

figure;
plot(t_eeg(idx), EEG(idx), 'b'); % Raw EEG [cite: 28, 29]
hold on;

% Draw vertical lines where spikes occurred
stem(cellTS(spike_idx), ones(sum(spike_idx),1) * max(EEG(idx)), 'r', 'Marker', 'none');
xlabel('Time (s)');
ylabel('Voltage');
title('EEG Theta and Spike Timing');