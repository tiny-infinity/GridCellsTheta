

import numpy as np
from scipy import signal
from scipy.signal import butter
import analysis_utils as a_utils

# np.set_printoptions(threshold=np.inf)


def calc_phase_stell(stell_spks, inst_phases, params):
    if params['vel_type'] == 'const':
        dc_in = np.max(params['stell_const_dc'])
    else:
        x = 0  # to be implemented

    separated_fields = a_utils.separate_fields(stell_spks)
    flattened_list = {key: [] for key in separated_fields.keys()}
    for i, cell in separated_fields.items():
        if cell != None:
            for field in cell[1:-1]:
                for spk in field:
                    flattened_list[i].append(spk)

    shifted_fields_d = a_utils.shift_fields_to_center(stell_spks)
    shifted_field_col = {}
    for key, val in shifted_fields_d.items():
        if val != None:
            fields = []
            for a_field in val:
                fields.extend(a_field)
            shifted_field_col[key] = fields
    stell_phases = {key: [] for key in separated_fields.keys()}
    for i, cell in shifted_field_col.items():
        stell_spikes_shifted = np.array(flattened_list[i])
        stell_spikes_shifted_idx = (stell_spikes_shifted/0.025).astype('int')
        # stell_phases[i]=inst_phases[stell_spikes_shifted_idx]%(2*np.pi)
        stell_phases[i] = (
            2*3.14*omega(dc_in)*(stell_spikes_shifted+params['phi_i_theta'])) % (2*np.pi)

    return shifted_field_col, stell_phases


def calc_phase_stell_avg(stell_spks, inst_phases, params):
    if params['vel_type'] == 'const':
        dc_in = max(params['stell_const_dc'])
    else:
        x = 0  # to be implemented
    separated_fields = a_utils.separate_fields(stell_spks)
    shifted_fields_d = a_utils.shift_fields_to_center(stell_spks)
    stell_phases = {key: [] for key in separated_fields.keys()}
    for i, cell in separated_fields.items():
        if cell != None:
            for field in cell[1:-1]:
                stell_spikes_ = np.array(field)
                stell_spikes_idx = (stell_spikes_/0.025).astype('int')
                # stell_phases[i].append(list(inst_phases[stell_spikes_idx]%(2*np.pi)))
                stell_phases[i].append(
                    (2*3.14*stell_spikes_*omega(dc_in)+1.57) % (4*np.pi))

    shifted_field_col = {}
    for key, val in shifted_fields_d.items():
        if val != None:
            fields = []
            for a_field in val:
                fields.extend(a_field)
            shifted_field_col[key] = fields

    return shifted_field_col, stell_phases


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfilt(sos, data)
    return y


def filter_signal(x, lowcut=5, highcut=12, fs=1/(0.025*1e-3), order=1):

    # Sample rate and desired cutoff frequencies (in Hz).
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order)
    return y


def omega_old(x):
    c0 = 1.09680758e-01
    c1 = 2.86083364e+02
    c2 = 3.65880847e+05
    c3 = 2.38144294e+08
    c4 = 7.65396313e+10
    c5 = 9.69300805e+12
    return (c0+c1*x+c2*(x**2)+c3*(x**3)+c4*(x**4)+c5*(x**5))-4.5e-4


def omega(x):
    return (8.04-0.1)*x+0.02556


def regression(stell_spk, stell_phase,):

    slopes = np.linspace(-5, 5, 10000)
    scores = np.zeros(10000)
    for k, val in enumerate(slopes):
        scores[k] = circ_regr(val, np.array(stell_spk) /
                              np.max(stell_spk), np.array(stell_phase))
    qhat = slopes[np.argmax(scores)]
    offset = np.arctan2(np.sum(np.sin(np.array(stell_phase)-2*np.pi*qhat*np.array(stell_spk)/np.max(stell_spk))),
                        np.sum(np.cos(np.array(stell_phase)-2*np.pi*qhat*np.array(stell_spk)/np.max(stell_spk))))
    stell_spks_norm = stell_spk/np.max(stell_spk)
    phi_bar = np.arctan2(np.sum(np.sin(stell_phase)),
                         np.sum(np.cos(stell_phase)))
    theta_j = (2*np.pi*abs(qhat)*stell_spks_norm) % (2*np.pi)
    theta_bar = np.arctan2(np.sum(np.sin(theta_j)), np.sum(np.cos(theta_j)))
    denom = np.sqrt(np.sum(np.sin(stell_phase-phi_bar)**2)
                    * np.sum(np.sin(theta_j-theta_bar)**2))
    corr_coef = np.sum(np.sin(stell_phase-phi_bar) *
                       np.sin(theta_j-theta_bar))/denom
    return corr_coef, qhat, np.max(scores), offset


def circ_regr(q, X, P):
    return np.sqrt(((np.sum(np.cos(P-2*np.pi*q*X))/len(X))**2)+((np.sum(np.sin(P-2*np.pi*q*X))/len(X))**2))
