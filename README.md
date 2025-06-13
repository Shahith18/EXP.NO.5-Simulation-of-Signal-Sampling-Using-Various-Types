# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling
    
# AIM
To perform experimental verification of signal sampling using various types.

# SOFTWARE REQUIRED
Google Colab

# ALGORITHMS
1.Import Libraries and Define Original Signal: Import necessary libraries: numpy and matplotlib.pyplot. Define original signal parameters: sampling frequency, time array, signal frequency, and signal amplitude.

2.Define Sampling Parameters: Define sampling frequency and time array for sampling the original signal.

3.Sample Original Signal: Sample the original signal using the defined sampling parameters to obtain the sampled signal.

4.Reconstruct Sampled Signal: Reconstruct the sampled signal using a reconstruction technique, such as zero-order hold or linear interpolation.

5.Plot Results: Plot the original signal, sampled signal, and reconstructed signal using matplotlib.pyplot to visualize the results.

# PROGRAM
i) Ideal Sampling
```
import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 100          # Sampling frequency (Hz)
f = 5             # Signal frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector

# Original signal (sine wave)
signal = np.sin(2 * np.pi * f * t)

# Plot original signal
plt.figure()
plt.plot(t, signal, label='Original Signal')
plt.title('Original Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Sampled signal (same as original because fs is sufficient)
plt.figure()
plt.plot(t, signal, 'b', alpha=0.5, label='Original')
plt.stem(t, signal, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled')
plt.title('Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Simulated reconstruction using linear interpolation
from numpy import interp
t_fine = np.linspace(0, 1, 1000)  # finer time grid for smooth line
reconstructed = interp(t_fine, t, signal)  # linear interpolation

# Plot reconstructed signal
plt.figure()
plt.plot(t, signal, 'b', alpha=0.5, label='Original (Sampled)')
plt.plot(t_fine, reconstructed, 'r--', label='Reconstructed (Interpolated)')
plt.title('Reconstructed Signal without SciPy')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
```
# OUTPUT
![impulse signal](https://github.com/user-attachments/assets/1ccec161-55db-4d0c-aedd-7de16a696bb2)



# PROGRAM
ii) Natural Sampling 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

### Parameters
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector

### Message Signal (sine wave message)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

### Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)

### Construct Pulse Train (rectangular pulses)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i+pulse_width] = 1  # Indentation fixed here

### Natural Sampling
nat_signal = message_signal * pulse_train

### Reconstruction (Demodulation) Process
sampled_signal = nat_signal[pulse_train == 1]

### Create a time vector for the sampled points
sample_times = t[pulse_train == 1]

### Interpolation - Zero-Order Hold (just for visualization)
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]

### Low-pass Filter (optional, smoother reconstruction)
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

plt.figure(figsize=(14, 10))

### Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

### Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

### Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)

### Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show
```
# OUTPUT
![image](https://github.com/user-attachments/assets/6effde83-b81e-4752-902d-ae9164d1be35)


# PROGRAM
iii) Flat Top Sampling

# PROGRAM
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
fs = 1000 # Sampling frequency (samples per second) T = 1 # Duration in seconds t = np.arange(0, T, 1/fs) # Time vector fm = 5 # Frequency of message signal (Hz) message_signal = np.sin(2 * np.pi * fm * t) pulse_rate = 50 # pulses per second pulse_train = np.zeros_like(t) pulse_width = int(fs / pulse_rate / 4) # Flat-top width for i in range(0, len(t), int(fs / pulse_rate)): pulse_train[i:i+pulse_width] = 1 flat_top_signal = np.copy(message_signal) for i in range(0, len(t), int(fs / pulse_rate)): flat_top_signal[i:i+pulse_width] =
message_signal[i] # Hold value constant sampled_signal = flat_top_signal[pulse_train == 1] sample_times = t[pulse_train == 1] reconstructed_signal = np.zeros_like(t) for i, time in enumerate(sample_times): index = np.argmin(np.abs(t - time)) reconstructed_signal[index:index+pulse_width] = sampled_signal[i] def lowpass_filter(signal, cutoff, fs, order=5): nyquist = 0.5 * fs normal_cutoff = cutoff / nyquist b, a = butter(order, normal_cutoff, btype='low', analog=False) return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs) plt.figure(figsize=(14, 10)) plt.subplot(4, 1, 1) plt.plot(t, message_signal, label='Original Message Signal') plt.legend() plt.grid(True) plt.subplot(4, 1, 2) plt.plot(t, pulse_train, label='Pulse Train') plt.legend() plt.grid(True) plt.subplot(4, 1, 3) plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal') plt.legend() plt.grid(True) plt.subplot(4, 1, 4) plt.plot(t, reconstructed_signal, label='Reconstructed Signal', color='green') plt.legend() plt.grid(True) plt.tight_layout() plt.show()
```
# OUTPUT
![image](https://github.com/user-attachments/assets/8a90dd1e-9311-454e-b50a-efa6b66833e2)


# RESULT / CONCLUSIONS
Thus the given eperiment ideal sampling ,natural sampling,flat top sampling has been verified successfully by using python
