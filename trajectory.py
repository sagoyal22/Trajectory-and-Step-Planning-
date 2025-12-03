import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and SORT the combined dataset
df = pd.read_csv("WALKING_AND_TURNING.csv")
print("Columns:", df.columns.tolist())

#  sort by timestamp so time is monotonic 
df = df.sort_values("timestamp").reset_index(drop=True)

t_ns = df["timestamp"].values
ax   = df["accel_x"].values
ay   = df["accel_y"].values
az   = df["accel_z"].values
gz   = df["gyro_z"].values

# time 
t = df["timestamp"].values
t = t - t[0]

fs = 1.0 / np.median(np.diff(t))
print(f"Sampling rate ≈ {fs:.2f} Hz")


# 2. Step detection 

acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

# EWMA smoothing
alpha_acc = 0.2
acc_smooth = np.zeros_like(acc_mag, dtype=float)
acc_smooth[0] = acc_mag[0]
for i in range(1, len(acc_mag)):
    acc_smooth[i] = alpha_acc * acc_mag[i] + (1 - alpha_acc) * acc_smooth[i - 1]

print("acc_smooth min / max:", float(acc_smooth.min()), float(acc_smooth.max()))

# choose a data-driven threshold: between median and high values
p50 = np.percentile(acc_smooth, 50)   # median
p90 = np.percentile(acc_smooth, 90)   # big peaks
step_threshold = 0.5 * (p50 + p90)

print("Step threshold on magnitude:", float(step_threshold))

min_step_time = 0.3  

steps = []
last_step_time = -np.inf

for i in range(1, len(acc_smooth) - 1):
    if (
        acc_smooth[i] > step_threshold and
        acc_smooth[i] > acc_smooth[i - 1] and
        acc_smooth[i] > acc_smooth[i + 1] and
        (t[i] - last_step_time) > min_step_time
    ):
        steps.append(i)
        last_step_time = t[i]

print(f"Detected steps: {len(steps)}")

#Debug plot:
plt.figure(figsize=(10, 4))
plt.plot(t, acc_smooth, label="Smoothed |accel|")
plt.axhline(step_threshold, color="orange", linestyle="--", label="Step threshold")
plt.scatter(t[steps], acc_smooth[steps], color="red", label="Detected steps")
plt.xlabel("Time (s)")     
plt.ylabel("Acceleration (m/s²)")
plt.title("Step detection on WALKING_AND_TURNING")
plt.legend()
plt.tight_layout()
plt.show()


# 3. Turn  estimation 

gx = df["gyro_x"].values
gy = df["gyro_y"].values
gz = df["gyro_z"].values

# Pick the axis with the largest variance 
std_x = np.std(gx)
std_y = np.std(gy)
std_z = np.std(gz)

if std_x >= std_y and std_x >= std_z:
    gyro_raw = gx
    print("Using gyro_x for heading")
elif std_y >= std_x and std_y >= std_z:
    gyro_raw = gy
    print("Using gyro_y for heading")
else:
    gyro_raw = gz
    print("Using gyro_z for heading")

# Smooth and integrate
alpha_gyro = 0.2
gyro_smooth = np.zeros_like(gyro_raw, dtype=float)
gyro_smooth[0] = gyro_raw[0]
for i in range(1, len(gyro_raw)):
    gyro_smooth[i] = alpha_gyro * gyro_raw[i] + (1 - alpha_gyro) * gyro_smooth[i - 1]

dt = np.diff(t)
dt = np.append(dt, dt[-1])          

theta = np.cumsum(gyro_smooth * dt) 

initial_heading_rad = np.deg2rad(90.0)
heading = theta - theta[0] + initial_heading_rad


# 4. Build 2D trajectory (1 m per detected step)
step_length = 1.0 

x, y = 0.0, 0.0
traj_x = [x]
traj_y = [y]

for idx in steps:
    ang = heading[idx]
    x += step_length * np.cos(ang)
    y += step_length * np.sin(ang)
    traj_x.append(x)
    traj_y.append(y)


plt.figure(figsize=(6, 6))
plt.plot(traj_x, traj_y, marker="o")
plt.scatter(traj_x[0],  traj_y[0],  color="green", label="Start")
plt.scatter(traj_x[-1], traj_y[-1], color="red",   label="End")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Reconstructed Walking Trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
