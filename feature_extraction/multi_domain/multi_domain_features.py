# -*- coding: utf-8 -*-
"""
Created on 2019/11/20 14:21

@author: John_Fengz
"""
import numpy as np
from scipy import fftpack, stats
import pywt


def feature_extraction(data, samping_rate):
    features = []
    """
    16 time-domain features, including mean_, mabs_, var_,
        rms_, max_, min_, ppv_, sra_, skew_v, kurtv_v, cf_,
        imp_f, mar_f, skew_f, kurtv_f, shp_f
    """
    # 1.1 Mean
    mean_ = np.mean(data)
    # 1.2 Absolute mean
    mabs_ = np.mean(np.abs(data))
    # 1.3 Variance
    var_ = np.var(data, ddof=1)
    # 1.4 Root mean variance
    rms_ = np.sqrt(np.mean(data * data))
    # 1.5 Maximum
    max_ = np.max(data)
    # 1.6 Minimum
    min_ = np.min(data)
    # 1.7 Peak-peak value
    ppv_ = max_ - min_
    # 1.8 Squre root of the ampliude
    sra_ = np.square(np.mean(np.sqrt(np.abs(data))))
    # 1.9 Skewness value
    skew_v = stats.skew(data)
    # 1.10 Kurtosis value
    kurtv_v = stats.kurtosis(data)
    # 1.11 Crest factor
    cf_ = np.max(np.abs(data)) / rms_
    # 1.12 Impulse factor
    imp_f = np.max(np.abs(data))/mabs_
    # 1.13 Margin factor
    mar_f = np.max(np.abs(data))/sra_
    # 1.14 Skewness factor
    skew_f = kurtv_v/np.power(var_, 3/2)
    # 1.15 Kurtosis factor
    kurtv_f = kurtv_v/np.power(rms_, 4)
    # 1.16 Shape factor
    shp_f = rms_/mabs_
    features_t = [mean_, mabs_, var_, rms_,
                  max_, min_, ppv_, sra_,
                  skew_v, kurtv_v, cf_, imp_f,
                  mar_f, skew_f, kurtv_f, shp_f]

    """
    12 frequency-domain features, including meanf, varf, maxf,
        minf, skewf, kurtf, fc, rmsf, stdf, cp1, cp2, cp3
     
    Ref: "Fault diagnosis in spur gears based on genetic
          algorithm and random forest"
    """
    # Transform original signal into frequency domain using FFT
    f_s = samping_rate  # Sampling rate
    n = len(data)
    freqs = fftpack.fftfreq(n, 1.0/f_s)
    freqs = freqs[range(int(n/2))]

    xfft = fftpack.fft(data)/n
    y = np.abs(xfft)[range(int(n/2))]

    # Mean
    meanf = np.sum(y) / len(y)
    # Variance
    varf = np.var(y, ddof=1)
    # Maximum
    maxf = np.max(y)
    # Minimum
    minf = np.min(y)
    # Skewness
    skewf = stats.skew(y)
    # Kurtosis
    kurtf = stats.kurtosis(y)
    # Frequency center
    fc = np.sum(freqs * y) / np.sum(y)
    # RMSF
    rmsf = np.sqrt(np.sum(np.square(freqs) * y) / np.sum(y))
    # STDF
    stdf = np.sqrt(np.sum(np.square(freqs-fc) * y) / np.sum(y))
    # CP1
    cp1 = np.sum(np.power(freqs-fc, 3) * y) / (len(y)*np.power(stdf, 3))
    # CP2
    cp2 = stdf / fc
    # CP3 This feature is changed because the CP3 in original paper
    # can not be calculated.
    cp3 = np.sum(np.power(freqs-fc, 4) * y) / (len(y)*np.power(stdf, 4))
    features_f = [meanf, varf, maxf, minf,
                  skewf, kurtf, fc, rmsf,
                  stdf, cp1, cp2, cp3]

    """
    32 features obtained by calculating the energy of each leaf 
    node wavelet packet
    
    Ref: "Heterogeneous Feature Models and Feature
          Selection Applied to Bearing Fault Diagnosis"
    """
    wp = pywt.WaveletPacket(data, 'db1', maxlevel=5)
    energy = []
    for node in wp.get_level(5, 'freq'):
        energy.append(np.linalg.norm(wp[node.path].data, ord=None))
    features_tf = energy / sum(energy)

    # Combine all features together
    features.extend(features_t)
    features.extend(features_f)
    features.extend(features_tf)

    return features
