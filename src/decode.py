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

import sys
import numpy as np
import scipy.io.wavfile as wf
from scipy.signal import windows
import watermark as wtrmrk

import erb_stuff as ef

eps = np.finfo(np.float).eps

def unpack_data_for_watermark(raw_bistream, nb_audio, wz_len):
    """
    Extract the Erb spectrums and positions from the watermark bistream
        INPUT:
            raw_bistream: the bitstream from the watermask
            nb_audio: number of audio sources
            wz_len: the number of Erb bands
        OUTPUT:
            m_A: 2d-array, the contritutions in stereo signals for each sources.
            m_ezq: 2d-array, the Erb spectrums for each sources
    """
    mark = wtrmrk.convert_bit_to_array(raw_bistream,\
                                        np.int8, \
                                        nb_audio * (wz_len - 1) + nb_audio)
    v_theta = mark[:nb_audio]     # get the position of sources
    m_A = np.vstack((np.sin(np.deg2rad(v_theta)), np.cos(np.deg2rad(v_theta))))

    m_ezq = mark[nb_audio:]     # get Erb spectrums
    m_ezq = m_ezq.reshape((nb_audio, (wz_len -1)))
    return m_A, m_ezq

def decode(wav_file, nb_audio, freq_cut, nfft, win_size, hop_size, M):
    """
    Extract the mono signals from the stereo signal
        INPUT:
            wav_files: list, mono audio filenames
            nb_audio: number of sources
            freq_cut: the maximun frequency for ERB-scale
            nfft: the number of ft bins
            win_size: size of squared Hann window (in samples)
                       it is also the processing frame size
            hop_size: overlape beetween frames (in samples)
            M : the sub-sample for ERB-scale
        OUTPUT:
            m_decoded_sigs : 2d-array, decoded signals
            sr: sample rate
    """

    # Load stereo signals
    sr, m_sigs = wf.read(wav_file)
    m_sigs = m_sigs.copy()
    m_sigs = m_sigs.T
    sigs_size = m_sigs.shape[1]

    # Compute Window function (sqrt of Hann)
    v_win = np.sqrt(windows.hann(win_size, sym=False))

    # 1/M ERB-scale
    v_wz = ef.compute_erbscale(nfft, sr, freq_cut, M)

    idx_f = 0
    pos = 0
    total_len = sigs_size - np.mod(sigs_size, win_size-hop_size)
    m_decoded_sigs = np.zeros((nb_audio, total_len))

    m_overlap_buff = np.zeros((nb_audio, hop_size))

    # Main loop, compute signals frame by frame
    while pos < (total_len - hop_size):
        print("Processing frame %04d" % (idx_f+1))
        cur_frame = m_sigs[:, pos:pos + win_size]

        tmp_frame = cur_frame.astype(np.int16)
        idx_beg = pos
        idx_end = min(pos+hop_size, total_len)

        # Extract watermark
        nb_total_ezq = ( nb_audio * (len(v_wz)-1) + nb_audio) * (8 * np.int8().itemsize)
        cur_ezq = np.zeros((2, int(nb_total_ezq/2)), dtype=np.int8)

        cur_ezq[0] = wtrmrk.decode_watermark(tmp_frame[0, :idx_end - idx_beg], B=3)[:int(nb_total_ezq / 2)]
        cur_ezq[1] = wtrmrk.decode_watermark(tmp_frame[1, :idx_end - idx_beg], B=3)[:int(nb_total_ezq / 2)]
        cur_ezq = np.concatenate((cur_ezq[0], cur_ezq[1]))      # concatenate Left and Right mark

        m_A, cur_ezq = unpack_data_for_watermark(cur_ezq, nb_audio, len(v_wz))

        m_decoded_sigs[:, idx_beg:idx_end], m_overlap_buff = decode_signal(cur_frame, v_wz, v_win, nfft,  m_A, cur_ezq, m_overlap_buff)

        idx_f += 1
        pos = pos + (hop_size)

    m_decoded_sigs = m_decoded_sigs.astype(np.int16)

    return m_decoded_sigs, sr



def decode_signal(m_sigs_x, v_esb_scale, v_win, nfft, m_A, m_ezq, m_overlap_buff):
    """
        Extract the mono frames from the stereo downmix frame
        INPUT:
            m_sigs_x: 2d-array, the stereo downmix signal
            v_erb_scale: 1d-array, the Erb scale
            v_win: 1d-array, the squared Hann window
            nfft: the number of ft bins
            m_A: 2d-array, the contritutions in stereo signals for each source
            m_eqz: 2d-array, the Erb spectrums for each source
            m_overlap_buff: 2d-array, the signals from previous frame
        OUTPUT:
            m_sigs_res: 2d-array, the sources signal
            m_overlap_buff: 2d-array, updated overlapp buff
    """
    nb_bits = 6
    m_ez = ef.unquant_erbenv(m_ezq, nb_bits)
    m_sigs_x =  m_sigs_x / 32767

    v_win_sigs_x = np.multiply(v_win[np.newaxis, :], m_sigs_x)
    tf_win_x = np.fft.fft(v_win_sigs_x, nfft)
    tf_win_x = tf_win_x[:, :int(nfft/2)+1]

    tf_win_y = wiener_filt(tf_win_x, m_ez, v_esb_scale, m_A)
    tf_win_y[:,[0, -1]] = np.real(tf_win_y[:,[0, -1]])
    tf_win_sigs = adjustenv(tf_win_y, m_ez, v_esb_scale)

    # Reconstruct the signal
    tf_win_sigs = np.concatenate((tf_win_sigs, \
                    np.conj(np.fliplr(tf_win_sigs[:, 1:int(nfft/2)]))), axis=1)
    m_sigs_sig =  np.fft.ifft(tf_win_sigs) * v_win[np.newaxis, :]
    m_sigs_sig = np.real(m_sigs_sig)      # make sure fourier transform is valid

    # Add overlappe from the previous frame
    m_sigs_res = m_overlap_buff + m_sigs_sig[:, 0 : m_overlap_buff.shape[1]]
    # Save the second part of signal into the overlappe buffer
    m_overlap_buff = m_sigs_sig[:, v_win.shape[0] - m_overlap_buff.shape[1]:]
    return m_sigs_res * 32767, m_overlap_buff


def wiener_filt(tf_x, m_ez, v_erb_scale, m_A):
    """
    Apply Wiener filter to extract the sources signals
        INPUT:
            tf_x: 2d-complex-array, the Fourier transform for stereo signals
            m_ez: 2d-array, the Erb spectrums for each sources
            v_erb_scale: 1d-array, the Erb scale
            m_A: 2d-array, the contritutions in stereo signals for each sources.
        OUTPUT:
            tf_y: 2d-complex-array, the approximation of Fourier transform for each source
    """
    nb_src = m_A.shape[1]
    nb_bins = tf_x.shape[1]
    tf_y = np.zeros((nb_src, nb_bins), dtype=np.complex)
    nb_bands = v_erb_scale.shape[0] - 1

    for idx_b in range(nb_bands):
        k = np.arange(v_erb_scale[idx_b], v_erb_scale[idx_b+1])
        k = k.astype(np.int)           # ft bins for specific band
        for idx_s in range(nb_src):
            alpha = np.multiply(m_A[:, idx_s], m_ez[idx_s, idx_b]**2) / \
                    (np.sum(np.multiply(m_A[:], m_ez[:, idx_b]**2), axis=1) + eps)
            tf_y[idx_s, k] = np.dot(alpha, tf_x[:, k])
    return tf_y

def adjustenv(tf_y, m_eq, v_erb_scale):
    """
    Adjust the Fourier transforms with the Erb enveloppes
        INPUT:
            tf_y: 2d-complex-array, the Fourier transform of each sources
            m_eq: 2d-array, Erb spectrum of each sources
            v_erb_scale: 1d-array, the Erb scale
        OUTPUT:
            tf_y: 2d-array, rectified Fourier transforms
    """
    nb_src = tf_y.shape[0]
    nb_bands = v_erb_scale.shape[0] - 1

    for idx_src in range(nb_src):
        for idx_b in range(nb_bands):
            k = np.arange(v_erb_scale[idx_b], v_erb_scale[idx_b+1])
            k = k.astype(np.int)                # ft bins for specific band
            m_mean = np.mean(np.power(np.abs(tf_y[idx_src, k]), 2))
            floor_value = max(m_mean, 1e-12)
            sigma = np.sqrt(m_eq[idx_src, idx_b] / floor_value)

            tf_y[idx_src, k] = tf_y[idx_src, k] * sigma

    return tf_y

def main():
    # Parameters
    win_size = 2**11
    hop_size = int(win_size / 2)
    nfft = win_size
    freq_cut = 22000
    M = 4

    if len(sys.argv) != 4:
        print("Usage, 3 parameters: stereo downmix filename, number of audio in downmix and prefix of output files")

    stereo_wavfile = sys.argv[1]
    nb_audio = int(sys.argv[2])
    unmix_prefix = sys.argv[3]

    m_decoded_sigs, sr = decode(stereo_wavfile, nb_audio, freq_cut,
                                nfft, win_size, hop_size, M)

    for idx_i in range(nb_audio):
        wf.write(unmix_prefix+"-"+str(idx_i)+".wav", sr, m_decoded_sigs[idx_i])

if __name__ == "__main__":
    main()
