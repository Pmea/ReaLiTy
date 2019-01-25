# //////////////////////////////////////////////////////////////////////////////
# Pierre Mahé (mahe.pierre@live.fr)
# L3I
# Université de La Rochelle
# 20-Jan-2019
#
# Based on idea from:
# Sylvain Marchand and Stanislaw Gorlow
# sylvain.marchand@univ-lr.fr and stanislaw.gorlow@labri.fr
# LaBRI CNRS
# Université Bordeaux 1
# 10-Jan-2012
# //////////////////////////////////////////////////////////////////////////////

import numpy as np

def compute_erbscale(nfft, sr, freq_cut, M):
    """
    Compute the erb scale filter
        INPUT:
            nfft: number of bin in Fourier Transform
            sr: sample rate
            freq_cut: the maximun frequency for Erb-scale
            M : the sub-sample for Erb-scale
        OUTPUT:
            m_wz, 2d-array, the Erb-scale index
            return te
    """
    erbrate = lambda f : np.max(21.4 * np.log10(4.37 * f +1), 0)
    erbrate = np.vectorize(erbrate)
    k_max = np.floor(freq_cut / sr * nfft) +1
    z_max = np.floor(erbrate(freq_cut / 1000))

    partition = np.arange(0, z_max + 2/M, 1/M)
    k = np.arange(1, k_max+1) / nfft * sr /1000
    z = erbrate(k)
    index = quantiz_erbscale(z, partition)

    m_wz = np.where(np.diff(index) != 0)[0]
    m_wz = np.concatenate((m_wz, np.array([k_max+1])))
    return m_wz

def quantiz_erbscale(v_sig, v_partition):
    """
        Quantified the erb-values into a partition.
        Return quantiz the Erb-scale (v_sig)
        INPUT:
            v_sig: 1d-array, the Erb-scale
            v_partition: 1d-array, linear mapping
        OUTPUT:
            idx: 1d-array, quantiz erb-scale
    """
    idx = np.zeros(v_sig.shape[0])
    for idx_p in range(v_partition.shape[0]):
        idx = idx + (v_sig > v_partition[idx_p])
    return idx

def compute_erbenv(tf_sigs, v_erb_scale):
    """
    Compute the Erb enveloppe of Fourier for each Fourier transform.
    Return the Erb enveloppes for each sources
        INPUT:
            tf_sigs: 2d-complex-array (nb_src, nfft)
                      Fourier transform for each sources (number of sources)
            v_erb_scale: 1d-array, the Erb mapping
        OUTPUT:
            m_e: 2d-array, the Erb spectrum for each fourier_transform signals
    """
    nb_src = tf_sigs.shape[0]
    Z = v_erb_scale.shape[0] -1
    m_e = np.zeros((nb_src, Z))

    for idx_s in range(nb_src):
        for idx_z in range(Z):
            k = np.arange(v_erb_scale[idx_z], v_erb_scale[idx_z+1])
            k = k.astype(np.int)
            m_e[idx_s, idx_z] = np.mean(np.power(np.abs(tf_sigs[idx_s, k]),2))
    return m_e

def quant_erbenv(m_ez, nb_bits):
    """
    Compress the Erb enveloppes
        INPUT:
            m_ez: 2d-array, Erb spectrum for each sources
            nb_bits: the number of bit for compression
        OUTPUT:
            m_ezq: 2d-array, quantiz Erb spectrum
    """
    m_ezq = np.round(1.5 * np.log2(m_ez))
    m_ezq = m_ezq + np.power(2, nb_bits)/2
    m_tmp = (m_ezq) > (np.power(2, nb_bits) -1)
    m_ezq[m_tmp] = np.power(2, nb_bits) -1

    m_tmp = (m_ezq) < 0
    m_ezq[m_tmp] = 0
    return m_ezq

def unquant_erbenv(m_ezq, nb_bits):
    """
    Restaure the Erb enveloppes values
        INPUT:
            m_ezq: 2d-array, quantiz Erb spectrum for each sources
                nb_bits: the number of bit used for compression
        OUTPUT:
            m_ez: 2d-array, Erb spectrum
    """
    return np.power( 2, (m_ezq - (2**nb_bits / 2 )) / 1.5)
