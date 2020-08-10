""" Code shell for HW 6, FFT problem.
Given:  tone_data (defines frequencies for digits),
        load_wav (loads a wav file)
"""
import numpy as np
import math
from numpy.fft import fft
from scipy.io import wavfile
from scipy.optimize import linprog  # for Q2


def tone_data():
    """ Builds the data for the phone number sounds
        Returns:
            tones - list of the freqs. present in the phone number sounds
            nums - a dictionary mapping the num. k to its two freqs.
            pairs - a dictionary mapping the two freqs. to the nums

        For example, 4 is represented by the two freqs 697 (low), 1336 (high)
        and nums[4] = (697, 1336)

        `pairs' maps the opposite way: pairs[(697, 1336)] = 4
    """
    lows = [697, 770, 852, 941]
    highs = [1209, 1336, 1477, 1633]  # (Hz)

    nums = {}
    for k in range(0, 3):
        nums[k+1] = (lows[k], highs[0])
        nums[k+4] = (lows[k], highs[1])
        nums[k+7] = (lows[k], highs[2])
    nums[0] = (lows[1], highs[3])

    pairs = {}
    for k, v in nums.items():
        pairs[(v[0], v[1])] = k

    return lows + highs, nums, pairs


def load_wav(fname):
    """ Loads a .wav file, returning the sound data.
        NOTE: rerns an Nxk array, where k = number of channels.
        (mono -> 1, stereo -> 2, etc.)

        Returns:
            rate - the sample rate (in samples/sec)
            data - an Nx1 (left/right) or Nx2 (stereo) np.array
                   of the samples.
            length - the duration of the sound (sec)
    """
    rate, data = wavfile.read(fname)
    if len(data.shape) > 1 and data.shape[1] > 1:
        print(f".wav file in stereo: returning {data.shape[1]} channels")
    length = data.shape[0] / rate
    print(f"Loaded sound file {fname}.")
    return rate, data, length

#----------------------------------------------------------------------
# Question 1
# Function to extract digit from tone's FFT

def read_digit(F, f):
    """ (a)  function to recognize a digit given freqs (f) and fft (F)"""
    tone_freq=f[np.nonzero(F>0.9*max(F))]
    tone_freq=tone_freq[tone_freq>0]
    freq, nums, pairs=tone_data()
    if (int(tone_freq[0]),int(tone_freq[1])) in pairs:
        digit=pairs.get((int(tone_freq[0]),int(tone_freq[1])))
    else:
        return None
    return digit

# Function to find out n-digit phone number from tone
def phone_number(fname):
    """ (b)  function to recognize a phone number given tone file(fname)"""
    rate, data, length = load_wav(fname)
    num_digits=round(length/0.7)
    num_samples=int(len(data)/num_digits)
    digits=''
    for i in range(0,len(data),num_samples):
        dig_data=data[i:i+num_samples]
        n=2**math.ceil(math.log2(abs(len(dig_data))))
        F=fft(dig_data,n)/len(dig_data)
        f=np.arange(-n/2,n/2)
        f=f*rate/n
        F=np.abs(np.roll(F,int(n/2)))
        digits=digits+str(read_digit(F,f))
    return digits

# Question-1 (b)-dial.wav : Ph. No. 5553429
ph_no_b1=phone_number('dial.wav')
print(ph_no_b1+' is the number from dial.wav.')
print('\n')
# Question-1 (b)-dial2.wav : Ph. No. 8006284
ph_no_b2=phone_number('dial2.wav')
print(ph_no_b2+' is the number from dial2.wav.')
print('\n')
# Question-1 (c)-noisy_dial.wav : Ph. No. 5553429
ph_no_c=phone_number('noisy_dial.wav')
print(ph_no_c+' is the number from noisy_dial.wav.')
print('\n')

#----------------------------------------------------------------------
# Question 2
def return_arrays():
    A=np.array([[2,1,1,0,0],[6,5,0,1,0],[2,5,0,0,1]])
    b=np.array([18,60,40])
    c=np.array([2,3,0,0,0])
    return A, b, c

slack=np.array([0,0,0])
A,b,c=return_arrays()
x=linprog(-c,A_eq=A,b_eq=b,method='revised simplex')
print('\n')
print(str(-x.fun)+' is the maximum value of the objective function and it occurs at x='+str(x.x[0])+' and y='+str(x.x[1])+'.')