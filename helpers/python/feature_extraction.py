import numpy as np
import scipy.stats as stats
from scipy.stats import entropy as shannon_entropy
from antropy import entropy, spectral_entropy, higuchi_fd
from pyentrp import entropy as ent
import spkit as sp
import nolds
import pandas as pd

def tsallis_entropy(signal, q=1.5, bins=50):
    # Discretize continuous signal into histogram
    hist, _ = np.histogram(signal, bins=bins, density=True)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    
    if np.isclose(q, 1.0):
        # Recover Shannon entropy
        return -np.sum(probs * np.log(probs))
    else:
        return (1.0 / (q - 1.0)) * (1 - np.sum(probs ** q))


def statistical_features(x):
    return {
        "Mean": np.mean(x),
        "Median": np.median(x),
        "Min": np.min(x),
        "Max": np.max(x),
        "Std": np.std(x),
        "RMS": np.sqrt(np.mean(x**2)),
        "Skewness": stats.skew(x),
        "Kurtosis": stats.kurtosis(x),
        "Peak": np.max(np.abs(x)),
        "IQR": np.percentile(x, 75) - np.percentile(x, 25),
        "Q1": np.percentile(x, 25),
        "Q2": np.percentile(x, 50),
        "Q3": np.percentile(x, 75)
    }

def entropy_features(x, sampling_frequency=256):  # sf = sampling frequency
    return {
        "Shannon_Entropy": sp.entropy(x, alpha=1),
        "Renyi_Entropy": sp.entropy(x, alpha=2),
        "Tsallis_Entropy": tsallis_entropy(x, q=1.2, bins=100),
        "Permutation_Entropy": entropy.perm_entropy(x, order=3, normalize=True),
        "Spectral_Flatness": entropy.spectral_entropy(x, sampling_frequency, method='fft'),
        "Log_Entropy": np.sum(np.log1p(np.abs(x))),
    }
    
def fractal_features(x):
    return {
        "Higuchi_FD": higuchi_fd(x),
        "Hurst_Exponent": nolds.hurst_rs(x),
    }

def hjorth_features(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x)
    var1 = np.var(dx)
    var2 = np.var(ddx)
    return {
        "Hjorth_Activity": var0,
        "Hjorth_Mobility": np.sqrt(var1 / var0),
        "Hjorth_Complexity": np.sqrt(var2 / var1) / np.sqrt(var1 / var0)
    }
    
def extract_all_features(x, sampling_frequency=256):
    # Ensure x is a contiguous 1D float64 array (safe for numba & libraries)
    x = np.ascontiguousarray(x, dtype=np.float64)

    feats = {}
    feats.update(statistical_features(x))
    feats.update(entropy_features(x, sampling_frequency))
    feats.update(fractal_features(x))
    feats.update(hjorth_features(x))
    return feats

def extract_org(signals, label, classification):
    feature_list = []
    for ch in range(signals.shape[1]):
        feats = extract_all_features(signals[:, ch])
        feats["Channel"] = ch+1
        feats["Mode"] = -1
        feats["Type"] = label
        feats["Class"] = classification
        feature_list.append(feats)
    return pd.DataFrame(feature_list)

def extract_decomp(signals, label, classification):
    feature_list = []
    for ch in range(signals.shape[0]):    
        for mode in range(signals.shape[2]):
            feats = extract_all_features(signals[ch, :, mode])
            feats["Channel"] = ch+1
            feats["Mode"] = mode+1
            feats["Type"] = label
            feats["Class"] = classification
            feature_list.append(feats)
    return pd.DataFrame(feature_list)