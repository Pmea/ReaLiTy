# //////////////////////////////////////////////////////////////////////////////
# Sylvain Marchand (sylvain.marchand@univ-lr.fr) and Pierre Mahé (mahe.pierre@live.fr)
# L3I
# Université de La Rochelle
# 20-Jan-2019
# //////////////////////////////////////////////////////////////////////////////

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from bitstream import BitStream

def convert_array_to_bit(v_values, values_type):
    """
        Convert an values-array with specific type to an binary array
        INPUT:
            v_values: 1d-array, values which be converted
            values_type: type of the v_values element.
                Usefull types; np.int8, np.int16, np.int32, np.float32
        OUTPUT:
            1d-array, return an integer array with binary values (0 or 1)
    """
    stream = BitStream()
    stream.write(v_values, values_type)
    bit_str = str(stream)
    return np.array(list(bit_str), dtype=int), len(v_values)

def convert_bit_to_array(v_bits, values_type, nb_values):
    """
        Convert a binary array into specific type array
        INPUT:
            v_bits: 1d-array,
            values_type: type of element in output array
            nb_values: number of elements to read
        OUTPUT:
            array with specific elements type indide
    """
    stream = BitStream()
    stream.write(v_bits.astype(bool), bool)
    return stream.read(values_type, nb_values)


def bitget(byteval, idx):
    """
        Return the bit value for specific position in byte
        INPUT :
            byteval: scalar,
            idx: scalar, index of wanted bit
        OUPUT :
            bit value for specific position
    """
    return ((byteval & (1 << idx)) != 0) * 1

def encode_watermark(src, mark, B):
    """
        Hidde the mark to data
        INPUT:
            src: 1d-array, data where the mark is hidde
            mark: 1d-array, the mark which must be hidde in src data.
                    Possible Mask values are only 0 or 1
            B: scalar, the number of bit used for hidde mark in each src data sample
                More bit here are, more mark data you can hidde, but src will be degrade
        OUPUT:
            dest : data with mark inside
    """
    # Mark processing
    # WARNING: mark and mask is differents things !
    # add zero-padding, if mark size is not a multiple of B
    mark = np.concatenate((mark, np.zeros(B - np.mod(mark.shape[0], B), dtype=mark.dtype)))
    mark = mark.reshape(int(mark.shape[0]/B), B).T
    mark = np.power(2, np.arange(B-1, -1, -1)).dot(mark)
    mask = np.power(2, B)-1 * np.ones(src.shape[0]).astype(np.int)
    m_shape = mark.shape

    # Add Mark to data
    dest = src
    dest[:m_shape[0]] = np.bitwise_or(src[:m_shape[0]], mask[:m_shape[0]]); # set all LSBs at 1
    dest[:m_shape[0]] = np.bitwise_xor(dest[:m_shape[0]], mark[:m_shape[0]]); # include mark (NOT) un LSBs
    dest[:m_shape[0]] = np.bitwise_xor(dest[:m_shape[0]], mask[:m_shape[0]]); # restore mark
    return dest

def decode_watermark(src, B):
    """
        Return the hidden informations from the data-src
        INPUT:
            src: 1d-array, the data with mark inside
            B: scalar, the number of bit used for hide mark in each src data sample
        OUTPUT:
            mark: 1d-array, the hidden informations
    """
    mark = np.zeros((B, src.shape[0]), dtype=int)
    for b in range(0, B):
        mark[b, :] = bitget(src, B-b-1)
    mark = mark.reshape((B*src.shape[0]), order='F')
    return mark

def main():
    values_type = np.int8
    mark, nb_values = convert_array_to_bit(np.arange(500, dtype=values_type), values_type)
    v_sig = np.round(np.random.rand(int(N/2)) * 64).astype(np.int16)

    v_coded_sig = encode_watermark(v_sig, mark, B=2)
    res_mask = decode_watermark(v_coded_sig, 2)
    res_mask = convert_bit_to_array(res_mask, values_type, nb_values)
    print(res_mask)

if __name__ == "__main__":
    main()
