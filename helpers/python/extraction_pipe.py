import os
import numpy as np
import pandas as pd

from .feature_extraction import extract_org, extract_decomp

def extract_info(eng, filepath, class_label):
    signal = eng.load_signal(filepath)

    # ------------------ Notch Filter ------------------
    signal = eng.notch_filter(signal, 128, 50, 30)

    # ------------------ Butterworth Filter ------------------
    signal = eng.butterworth_filter(signal, 128, 0.1, 60, 6)

    # ------------------ EMD ------------------
    emd_signals = []
    for ch in range(1, 20):
        emd_signal = eng.empirical_mode_decomposition(signal, ch, 6)
        emd_signals.append(np.array(emd_signal))

    # ------------------ EWT ------------------
    ewt_signals = []
    for ch in range(1, 20):
        ewt_signal = eng.empirical_wavelet_transformation(signal, ch, 6)
        ewt_signals.append(np.array(ewt_signal))

    # ------------------ VMD ------------------
    vmd_signals = []
    for ch in range(1, 20):
        vmd_signal = eng.variational_mode_decomposition(signal, ch, 6)
        vmd_signals.append(np.array(vmd_signal))

    # Extract features
    df_original = extract_org(np.array(signal), "Original", class_label)
    df_emd = extract_decomp(np.array(emd_signals), "EMD", class_label)
    df_ewt = extract_decomp(np.array(ewt_signals), "EWT", class_label)
    df_vmd = extract_decomp(np.array(vmd_signals), "VMD", class_label)

    final_features = pd.concat([df_original, df_emd, df_ewt, df_vmd])
    return final_features