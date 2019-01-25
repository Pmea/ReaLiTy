# //////////////////////////////////////////////////////////////////////////////
# Pierre Mahé (mahe.pierre@live.fr)
# L3I
# Université de La Rochelle
# 20-Jan-2019
# //////////////////////////////////////////////////////////////////////////////

import os
import numpy as np


regex_wavfile = "examples-audio/example2_\*.wav" #"examples-audio/example1_\*.wav"
out_wavfile = "res-audio/out.wav"
unmix_prefix = "res-audio/unmix"

nb_audio = 5
v_theta = (np.arange(nb_audio)+1) / (nb_audio+1) * 90  #homogenous source distribution
v_theta = v_theta.astype(str)
v_theta = " ".join(v_theta)


os.system("python3.6 encode.py "+ regex_wavfile + " " + str(nb_audio) + " " + out_wavfile + " " + v_theta)
os.system("python3.6 decode.py "+ out_wavfile + " " + str(nb_audio) + " " + unmix_prefix)
