
# %% Cell 1: Imports

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# %% Cell 2: Generate the signal

t = torch.linspace(0, 4*torch.pi, 1000)

amplitude_fundamental = 1
phase = 0
frequency_fundamental = 1

sinusoid_1 = amplitude_fundamental * torch.sin(frequency_fundamental*t + phase)

amplitude_second_harmonic = 0.5
freq_second_harmonic = 2

sinusoid_2 = amplitude_second_harmonic * torch.sin(freq_second_harmonic*t + phase)

combined_signal = sinusoid_1 + sinusoid_2  # Original, clean signal

noise = 0.1 * torch.randn_like(combined_signal)

noisy_signal = combined_signal + noise  # Signal with added noise


# %% Cell 3: Plot original and noisy signals

plt.figure(figsize=(12, 5))

plt.plot(t.numpy(), combined_signal.numpy(),
         label="Original Signal")

plt.plot(t.numpy(), noisy_signal.numpy(),
         label="Noisy Signal", alpha=0.7)

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Original Signal vs Gaussian White Noise")
plt.legend()
plt.grid()

plt.show()


# %% Cell 4: Define the Gaussian function

# Write a function to fit the signals to a Gaussian.
# x has to be the first argument and the function parameters next.

def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))


# %% Cell 5: Define the sum of Gaussians

def gaussian_sum(x, *params):
    total_gaussian = np.zeros_like(x)

    for i in range(0, len(params), 3):
        amplitude = params[i]
        mean = params[i+1]
        std_dev = params[i+2]

        total_gaussian += gaussian(x, amplitude, mean, std_dev)

    return total_gaussian


# %% Cell 6: Find the peaks in the noisy signal

peak_indices, _ = find_peaks(noisy_signal.numpy(), prominence=0.4)

# Find the time and amplitude of each detected peak
time_peaks = t[peak_indices].numpy()
peak_amplitudes = noisy_signal[peak_indices].numpy()

print("Time peaks:", time_peaks)
print("Shape:", time_peaks.shape)

#  highlight detected peaks by red dots 

plt.figure(figsize=(12, 5))

plt.plot(t.numpy(), noisy_signal.numpy(), label="Noisy Signal")

# Convert signal to NumPy
y = noisy_signal.numpy()
x = t.numpy()

# Detect positive peaks
positive_peak_indices, _ = find_peaks(y, prominence=0.4)

# Detect negative peaks (troughs)
negative_peak_indices, _ = find_peaks(-y, prominence=0.4)

# Combine and sort all peak indices
peak_indices = np.sort(
    np.concatenate((positive_peak_indices, negative_peak_indices))
)

time_peaks = x[peak_indices]
peak_amplitudes = y[peak_indices]

print("Positive peaks:", len(positive_peak_indices))
print("Negative peaks:", len(negative_peak_indices))
print("Total peaks:", len(peak_indices))

plt.figure(figsize=(12, 5))

plt.plot(x, y, label="Noisy Signal")

plt.scatter(
    x[positive_peak_indices],
    y[positive_peak_indices],
    color="red",
    label="Positive Peaks"
)

plt.scatter(
    x[negative_peak_indices],
    y[negative_peak_indices],
    color="green",
    label="Negative Peaks"
)

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Peak Detection on Noisy Signal")
plt.legend()
plt.grid()
plt.show()


# %% Cell 7: Formulate the initial guess
K= 3
initial_guess = []

for i in range(len(peak_indices)):
    peak_time=time_peaks[i]
    peak_amplitude= peak_amplitudes[i]
    if K==1:
       offsets= np.array([0])
    else:
         offsets= np.linspace(-0.25, 0.25, K)
    initial_std= 0.55
    normalization = np.sum(
        np.exp(-(offsets**2) / (2 * initial_std**2))
    )
    initial_amplitude = peak_amplitude / normalization
    for k in range(K):
        amplitude= initial_amplitude
        mean= peak_time+ offsets[k]
        st_dev= initial_std
        initial_guess.extend([
            amplitude,
            mean,
            st_dev
        ])
print("Number of Gaussians:", len(initial_guess) // 3)
print("Number of parameters:", len(initial_guess))
print(np.array(initial_guess).reshape(-1, 3))


# %% Cell 8: Fit the Gaussians

lower_bounds = []
upper_bounds = []

for i in range(len(peak_indices)):

    peak_time = time_peaks[i]
    peak_amplitude = peak_amplitudes[i]

    for k in range(K):

        # Amplitude bounds depend on whether the peak is positive or negative
        if peak_amplitude >= 0:
            lower_bounds.extend([0, max(0, peak_time - 0.65), 0.08])
            upper_bounds.extend([2.5, min(4*np.pi, peak_time + 0.65), 1.5])

        else:
            lower_bounds.extend([-2.5, max(0, peak_time - 0.65), 0.08])
            upper_bounds.extend([0, min(4*np.pi, peak_time + 0.65), 1.5])

popt, pcov = curve_fit(
    gaussian_sum,
    t.numpy(),
    noisy_signal.numpy(),
    p0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    max_nfev=100000,
    x_scale='jac'
)

print("Fitted parameters:")
print(popt.reshape(-1, 3))
# %% Cell 9: Calculate fitted signal
# %% Cell 9: Calculate fitted signal

fitted_y = gaussian_sum(t.numpy(), *popt)

# Error against noisy signal
rmse_noisy = np.sqrt(
    np.mean((noisy_signal.numpy() - fitted_y)**2)
)

# Error against original clean signal
rmse_clean = np.sqrt(
    np.mean((combined_signal.numpy() - fitted_y)**2)
)

print("RMSE against noisy signal:", rmse_noisy)
print("RMSE against clean signal:", rmse_clean)


# %% Cell 10: Plot the Gaussian fit

# %% Cell 10: Plot Gaussian fit

plt.figure(figsize=(12, 5))

plt.plot(
    t.numpy(),
    combined_signal.numpy(),
    label="Original Clean Signal",
    color="blue"
)

plt.plot(
    t.numpy(),
    noisy_signal.numpy(),
    label="Noisy Signal",
    color="gray",
    alpha=0.5
)

plt.plot(
    t.numpy(),
    fitted_y,
    label="Gaussian Fit",
    color="red",
    linestyle="--",
    linewidth=2
)

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title(f"Gaussian Fit with K={K} Gaussians per Peak")
plt.legend()
plt.grid()
plt.show()

# %% Cell 11: Torch version of the Gaussian calculation

# popt is a flat list: [amp, mean, std, amp, mean, std, ...]
params = torch.from_numpy(popt.reshape(-1, 3))

amplitude = params[:, 0]
mu = params[:, 1]
sigma = params[:, 2]
grid = t

gaussians = amplitude[None, :] * torch.exp(
    -(grid[:, None] - mu[None, :])**2
    / (2 * sigma[None, :]**2)
)

combined_gaussian = gaussians.sum(dim=1)

print(gaussians.shape)
print(combined_gaussian.shape)




# %%
