import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "TURNING.csv",
    usecols=[
        "timestamp",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z",
    ]
)

print("Rows in TURNING.csv:", len(df))
print("Columns in TURNING.csv:", df.columns.tolist())

# Extract gyroscope data
t_ns = df["timestamp"].values        
g_x  = df["gyro_x"].values
g_y  = df["gyro_y"].values
g_z  = df["gyro_z"].values

# Convert timestamps
t = (t_ns - t_ns[0]) / 1e9           


gyro_raw = g_z

# EWMA smoothing

alpha = 0.2
gyro_smooth = np.zeros_like(gyro_raw, dtype=float)
gyro_smooth[0] = gyro_raw[0]

for i in range(1, len(gyro_raw)):
    gyro_smooth[i] = alpha * gyro_raw[i] + (1 - alpha) * gyro_smooth[i - 1]

plt.figure(figsize=(10, 4))
plt.plot(t, gyro_raw, label="Raw gyro_z (rad/s)", alpha=0.5)
plt.plot(t, gyro_smooth, label="Smoothed gyro_z (EWMA)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Raw vs Smoothed Gyroscope (z-axis)")
plt.legend()
plt.tight_layout()
plt.show()



dt = np.diff(t)
dt = np.append(dt, dt[-1])  
theta = np.cumsum(gyro_smooth * dt)   
theta_deg = np.rad2deg(theta)

# Turn detection 

turn_indices = []
turn_deltas = []


prev_step = int(round(theta_deg[0] / 90.0))
prev_angle = theta[0]   

for i in range(1, len(theta_deg)):
    current_step = int(round(theta_deg[i] / 90.0))

    if current_step != prev_step:
       
        turn_indices.append(i)
        delta = theta[i] - prev_angle  
        turn_deltas.append(delta)

        prev_step = current_step
        prev_angle = theta[i]

# Summarize results
print(f"Total turns detected: {len(turn_indices)}")
for idx, delta in zip(turn_indices, turn_deltas):
    angle_deg_change = np.rad2deg(delta)
    direction = "counter-clockwise" if angle_deg_change > 0 else "clockwise"
    print(
        f"Turn at t = {t[idx]:.2f}s: "
        f"Δθ ≈ {angle_deg_change:.1f}°, direction = {direction}"
    )



plt.figure(figsize=(10, 4))
plt.plot(t, theta_deg, label="Orientation angle (deg)")
plt.scatter(
    t[turn_indices],
    theta_deg[turn_indices],
    color="red",
    label="Detected turns",
)
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Detected 90° Turns from Gyroscope Data")
plt.legend()
plt.tight_layout()
plt.show()
