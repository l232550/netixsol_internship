# Task:
# Given a raw NumPy array simulating sensor data (provide a small synthetic dataset),
# compute rolling statistics, normalize the data (z-score), and flag outliers (>2 std dev)
# — no Pandas allowed, NumPy only.

import numpy as np

# I have been feeling really excited about Remote Sensing lately so I kinda wanna use a related dataset
# sure I can : )

# this is my scenario for the synthetic dataset:

'''A Sentinel-2 satellite is monitoring the surface temperature of a region over multiple observation dates.
 One reading is unusually high, possibly indicating a heat event or a sensor anomaly.'''

temperature = np.array([
    29.1, 29.4, 29.3, 29.5, 29.7,
    30.2, 29.8, 41.2, 30.1, 29.9
])

print("The synthetic dataset values for Sentinel-2 surface temperature:\n", temperature)

# now let's calculate rolling statistics, i.e., mean, std, z-score and outlier detection but with a window

'''normally, in maths:
mean = sum/n
std = sqrt(sum(x_i - mean)**2/n-1) in loop
z-score = (x - mean)/std


but in numpy, we have:
mean_of_data = np.mean(temperature)
std_of_data = np.std(temperature) # uses population std, divides by n

for sample std (which divides by n-1): np.std(temperature, ddof=1)

numpy doesn't have built-in function for z-score
z_scores = (temperature - mean_of_data) / std_of_data
'''

window = 3  # for rolling stats, let's consider a window of 3

rolling_means = []
rolling_stds = []

for i in range(len(temperature) - window + 1):

    current_window = temperature[i:i+window]

    mean_of_window = np.mean(current_window)
    std_of_window = np.std(current_window)

    rolling_means.append(mean_of_window)
    rolling_stds.append(std_of_window)

rolling_means = np.round(rolling_means, 1)
rolling_stds = np.round(rolling_stds, 1)

print("\nRolling means:")
print(rolling_means)

print("\nRolling standard deviations:")
print(rolling_stds)

# normalize the entire dataset using z-score

mean_of_data = np.mean(temperature)
std_of_data = np.std(temperature)

z_scores = (temperature - mean_of_data) / std_of_data

print("\nZ-scores:")
print(z_scores)

# flag outliers (> 2 standard deviations away from mean)

outliers = np.abs(z_scores) > 2

print("\nOutlier flags:")
print(outliers)

print("\nOutlier value(s):")
print(temperature[outliers])

print("\nOutlier details:")

for i in range(len(temperature)):
    if outliers[i]:
        print(
            f"Temperature = {temperature[i]}°C, "
            f"Z-score = {z_scores[i]:.2f}"
        )