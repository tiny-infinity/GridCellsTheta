import importlib
from neuron import h
import interneuron_amb
import matplotlib.pyplot as plt
import numpy as np
importlib.reload(interneuron_amb)

h.load_file("stdrun.hoc")
h.celsius = 37

test_intrnrn = interneuron_amb.Interneuron(1)
theta_ic = h.IClamp(test_intrnrn.soma(0.5))
theta_ic.dur = 1e9
freq = 10
time_ms = 1000
osc_amp = 10e-4
baseline_amp = 0.8e-3
T, num_steps = time_ms/1000,time_ms
time_arr = np.linspace(0,T,num_steps)
curr_arr = osc_amp * np.sin(2 * np.pi * freq * time_arr) + np.full_like(time_arr, baseline_amp)
print((time_arr))
curr_vec = h.Vector(curr_arr)
curr_vec.play(theta_ic._ref_amp, True)

spike_times_intrnrn = h.Vector()

nc_intrnrn = h.NetCon(test_intrnrn.soma(0.5)._ref_v, None, sec=test_intrnrn.soma)
nc_intrnrn.threshold = 0
nc_intrnrn.record(spike_times_intrnrn)

i_theta = h.Vector().record(theta_ic._ref_amp)

test_intrnrn_v = h.Vector().record(test_intrnrn.soma(0.5)._ref_v)
time = h.Vector().record(h._ref_t)

h.finitialize(-65)
h.continuerun(1000)
fig,ax = plt.subplots(figsize=(10,6))
ax1 = ax.twinx()
ax.plot(time, i_theta.to_python(), color='gray', linestyle='--', label='Theta Input')
ax1.plot(time, test_intrnrn_v, label='Interneuron')
plt.show()