# Underdetermined Source Signal Recovery
##### Last update: 25 Jan 2019
****

This algorithm is inpired by works done during *DReaM* project and specialy the *Underdetermined Source Signal Recovery* (USSR) algorithm from Sylvain Marchand ([sylvain.marchand@univ-lr.fr](mailto:sylvain.marchand@univ-lr.fr)) and Stanislaw Gorlow.
Algorithm and implementation are writed by Pierre Mahe ([mahe.pierre@live.fr](mailto:mahe.pierre@live.fr)) with  Sylvain Marchand helps.

```The algorithm implementation is licensed under GNU GPL v3```

## Description
Le goal of the algorithm is to take mono multi-tracks and with those audios create a Stereo downmix. In this stereo mix each audio source have arbitrary position (intensity panning).
A Watermark is add to the stereo mix to provide enough informations to extract the mono-tracks from the stereo mix. The coding/decoding is not allow perfect recontruction but the audio quality is near the orignal one.

The watermarking algorithm uses Least Significant Bit coding. The upmix informations are hide in least significant bits.  
In the decode phase, the algorithm uses Wiener filter and upmix informations to reconstruct the mono-sources from stereo downmix.


#### Maximun sources in Stereo Mix
The number of sources do not have arbitrary limit , the limit is set by the number of bit used for Watermark. 
More bits are allocated in each sample, more informations you have but more noises are added to the stereo downmix signal.

#### How to determine number of bits needed for watermark ?
The quantity of data needed for upmix is calculed with this formula:
```([#audio] * [# ERB filter] + [#audio position]) * [number of bit to code varible]```

*#* symbole denote *cardinal* opperation.

The quantity of data whose can be hidden in watermark. For both stereo channels:
```
	2 * ([# fft bins]/2 * [number allocated of watermark])
```

**Example**: 5 input tracks, with 137 ERB filters, 2048 ft bin and watermark bit is 3.  
Data needed for upmix: ( 5 * 137 + 5) * 8 = 5520.   
Maximun data for watermark: 2 * ((2048 / 2) * 3) = 6144.  
This number of bits works !

## Caution
* All input tracks need to have the same length.
* The Stereo downmix do not be satured. If it's the case, decrease loudness of input tracks


## Installation

The algorithm need Python3 (3.6 or higher)

To install dependancies, executing this command in project directiory :
```
pip3.6 install -r requirements.txt 
```

## Suggested Improvement:

In low frequencies, audio quality is not as good as than in medium and high frequencies. Usage of CQT transforme may improve quality.