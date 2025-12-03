import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("WALKING.csv")

t = df["timestamp"].values
ax = df["accel_x"].values
ay = df["accel_y"].values
az = df["accel_z"].values

# Convert time to seconds
t = (t - t[0]) / 1e9


# Acceleration magnitude

acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

# EWMA smoothing 

alpha = 0.2
smooth = np.zeros_like(acc_mag)
smooth[0] = acc_mag[0]

for i in range(1, len(acc_mag)):
    smooth[i] = alpha * acc_mag[i] + (1 - alpha) * smooth[i - 1]


# Step detection

threshold = 13.0            
min_step_time = 0.4        

steps = []
last_step_time = -np.inf

for i in range(1, len(smooth) - 1):
    if (
        smooth[i] > threshold and
        smooth[i] > smooth[i - 1] and
        smooth[i] > smooth[i + 1] and
        (t[i] - last_step_time) > min_step_time
    ):
        steps.append(i)
        last_step_time = t[i]

print(f"Detected number of steps: {len(steps)}")


# Plot raw vs smoothed

plt.figure(figsize=(10, 4))
plt.plot(t, acc_mag, label="Raw acceleration magnitude", alpha=0.5)
plt.plot(t, smooth, label="Smoothed acceleration (EWMA)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Raw vs Smoothed Acceleration Magnitude")
plt.legend()
plt.tight_layout()
plt.show()

# Plot detected steps

plt.figure(figsize=(10, 4))
plt.plot(t, smooth, label="Smoothed acceleration")
plt.scatter(t[steps], smooth[steps], color="red", label="Detected steps")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Detected Steps")
plt.legend()
plt.tight_layout()
plt.show()
