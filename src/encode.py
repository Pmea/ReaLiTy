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

import glob
import sys
import numpy as np
import scipy as scp
import scipy.io.wavfile as wf
from scipy.signal import windows
import watermark as wtrmrk

import erb_stuff as ef

eps = np.finfo(np.float).eps

def pack_data_for_watermark(nb_audio, v_theta, m_ezq):
    """
    Pack the Erb spectrums and positions and convert it to bitstream
        INPUT:
            nb_audio: number of audio sources
            v_theta: 1d-array, the positions of sources
            m_ezq: 2d-array, the Erb-spectrums sources
        OUTPUT:
            mask, 2d-array, the watermark bitstream for each stereo signals
            nb_val, number of elements in watermark
    """
    # concatenate erb-spectrum to an 1d-array
    m_ezq = m_ezq.astype(np.int8)
    m_ezq = m_ezq.reshape(m_ezq.shape[1] * nb_audio)

    #Concatenate v_theta and erb data
    v_theta = v_theta.astype(np.int8)
    m_ezq = np.concatenate((v_theta, m_ezq))

    # Create bitsteam from the integer array
    mask, nb_val = wtrmrk.convert_array_to_bit(m_ezq, np.int8)
    # Split bitstream to the two stereo channels
    mask = mask.reshape((2, int(mask.shape[0]/2)))
    return mask, nb_val

def encode(wav_files, freq_cut, nfft, win_size, hop_size, M, v_theta):
    """
    Encode stereo downmix with watermarkself. Create a downmix with
    the angles position (v_theta). Erb-spectrums are hidden in stereo downmix.
        INPUT:
            wav_files: list, mono audio filenames
            freq_cut: the maximun frequency for ERB-scale
            nfft: the number of fft bins
            win_size: size of squared Hann window (in samples)
                       it is also the processing frame size
            hop_size: overlape beetween frames (in samples)
            M : the sub-sample for ERB-scale
            v_theta: 1d-array, the position of each source (in degree)
        OUTPUT:
            m_coded_sigs: 2d-array, the downmix stereo signal
            sr: sample rate
    """
    # Load source signals
    nb_audio = len(wav_files)

    m_sigs = np.empty(nb_audio, dtype=object)
    for audio_file, idx_f in zip(wav_files, range(nb_audio)):
        sr,  m_sigs[idx_f] = wf.read(audio_file)
        if m_sigs[idx_f].dtype == np.float32:   #If audio is coded in float-point wav
            m_sigs[idx_f] = m_sigs[idx_f].copy() * 32767
    m_sigs = np.stack(m_sigs)
    sigs_size = m_sigs.shape[-1]

    # Compute Window function (sqrted Hann)
    v_win = np.sqrt(windows.hann(win_size, sym=False))

    # 1/M ERB-scale
    v_wz = ef.compute_erbscale(nfft, sr, freq_cut, M)

    # Mixing matrix
    v_theta = v_theta.astype(np.int8)
    assert((np.unique(v_theta).shape[0]) == nb_audio)
    m_A = np.vstack((np.sin(np.deg2rad(v_theta)), np.cos(np.deg2rad(v_theta))))

    # Zeros-padding the signals to do not lost signals informations
    m_sigs = np.concatenate((np.zeros((nb_audio, hop_size), dtype=np.int16), \
                             m_sigs, \
                             np.zeros((nb_audio, win_size - hop_size - np.mod(sigs_size, win_size-hop_size)), dtype=np.int16)), \
                            axis=1)

    # Loop initialisation
    idx_f = 0
    pos = 0
    total_len = m_sigs.shape[1]
    m_coded_sigs = np.zeros((2, total_len), dtype=np.int16) #Output stereo signal

    # Main loop, compute signals frame by frame
    while pos < (total_len - hop_size):
        print("Processing frame %04d" % (idx_f+1))

        cur_frame = m_sigs[:, pos:pos + win_size]

        # Compute stereo downmix
        m_coded_frame = np.dot(m_A, cur_frame)
        m_coded_frame = m_coded_frame.astype(np.int16)

        # Compute the Erb spectrum for each mono-signal
        v_ezq = encode_signal(cur_frame, v_wz, v_win, nfft)

        # Add watermark
        idx_beg = pos
        idx_end = min(pos+hop_size, total_len)

        mask, nb_val = pack_data_for_watermark(nb_audio, v_theta, v_ezq)
        m_coded_sigs[0, idx_beg:idx_end] = wtrmrk.encode_watermark(m_coded_frame[0, :idx_end-idx_beg], mask[0], B=3)
        m_coded_sigs[1, idx_beg:idx_end] = wtrmrk.encode_watermark(m_coded_frame[1, :idx_end-idx_beg], mask[1], B=3)

        idx_f += 1
        pos = pos + (hop_size)

    return m_coded_sigs.T, sr

def encode_signal(m_sig, v_erb, v_win, nfft):
    """
    Compute the Erb spectrums for each sources
        INPUT:
            m_sig: 2d-array, the sources signals
            v_erb: 1d-array, the Erb scale
            v_win: 1d-array, the squared Hann window
            nfft: the number of ft bins
        OUTPUT:
            m_ezq: the Erb spectrums of each sources
    """
    nb_bits = 6
    m_sig = m_sig / 32767.
    m_win_sig = np.multiply(v_win[np.newaxis, :], m_sig)      # apply window
    tf_sigs = np.fft.fft(m_win_sig, nfft)   # compute fourier transforms
    tf_sigs = tf_sigs[:, :int(nfft/2)+1]

    m_ez = ef.compute_erbenv(tf_sigs, v_erb)
    m_eqz = ef.quant_erbenv(m_ez, nb_bits)

    return m_eqz

def main():
    # Parameters
    win_size = 2**11
    hop_size = int(win_size / 2)
    nfft = win_size
    freq_cut = 22000
    M = 4

    regex_wavfile = sys.argv[1]
    nb_audio = int(sys.argv[2])
    stereo_wavfile = sys.argv[3]
    v_theta = np.array(sys.argv[4:])
    v_theta = v_theta.astype(np.float)

    # Audio mono file
    # All files must have the same length
    wav_files = glob.glob(regex_wavfile)
    wav_files.sort()

    m_coded_sigs, sr = encode(wav_files, freq_cut, nfft, win_size, hop_size, M, v_theta)

    # Write stereo downmix
    wf.write(stereo_wavfile, sr, m_coded_sigs)


if __name__ == "__main__":
    main()
