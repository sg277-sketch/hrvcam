#%% 
import torch 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
#%%
t= torch.linspace(0,4*torch.pi,1000)
amplitude_fundamental= 1
phase= 0
frequency_fundamental=1 
sinusoid_1= amplitude_fundamental*torch.sin(frequency_fundamental*t+phase)
amplitude_second_harmonic=0.5
freq_second_harmonic=2
sinusoid_2= amplitude_second_harmonic*torch.sin(freq_second_harmonic*t+phase)
combined_signal=sinusoid_1+sinusoid_2
plt.plot(t.numpy(), combined_signal.numpy())

#write a function to fit the signals to a gaussian, x has to be the first argument and the func parameters next
def gaussian(x, amplitude, mean, std_dev):
  return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))

#formulate an inital guess for the parameters of the gaussian fit 
peak_amplitude= torch.max(combined_signal).item()
#use torch.argmax to find index of peak 
peak_index= torch.argmax(combined_signal)
#use the index to find the time of the peak

def gaussian_sum(x,*params):
    total_gaussian= np.zeros_like(x)
    for i in range(0, len(params),3):
        amplitude=params[i]
        mean=params[i+1]
        std_dev=params[i+2]
        total_gaussian+=gaussian(x,amplitude,mean,std_dev)
    return total_gaussian

peak_indices,_= find_peaks(combined_signal.numpy())
#t is the variable where we stored all our time points so it uses the index to find the time of the peak
time_peaks = t[peak_indices].numpy().reshape(-1)
print("Time peaks:", time_peaks)
print("Shape:", time_peaks.shape)
peak_amplitudes= combined_signal[peak_indices].numpy()
initial_guess=[]
for i in range(len(peak_indices)):
   initial_guess.extend([peak_amplitudes[i], time_peaks[i],0.5])
popt, pcov= curve_fit(gaussian_sum, t.numpy(), combined_signal.numpy(), p0= initial_guess)
print(popt)
print("Fitted parameters:")
print(popt.reshape(-1, 3))
fitted_y = gaussian_sum(t.numpy(), *popt)
plt.figure()
plt.plot(t.numpy(), combined_signal.numpy(), label='Original Signal', color='blue')
plt.plot(t.numpy(), fitted_y, label='Gaussian Fit', color='red', linestyle='--')
plt.legend()
plt.show()
amplitude = torch.tensor([1.0, -1.0, 0.5])
mu = torch.tensor([2.0, 3.0, 1.0])
sigma = torch.tensor([0.5, 0.5, 0.5])
gaussians = amplitude * torch.exp(-(grid[:, None] - mu[None, :])**2) /(2 * sigma[None, :]**2)
print(gaussians.shape)
print(combined_gaussian.shape) 
