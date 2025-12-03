import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("ACCELERATION.csv")

t = df["timestamp"].values                  
a_true = df["acceleration"].values            
a_noisy = df["noisyacceleration"].values     

t = t - t[0]

dt = np.diff(t)
dt = np.append(dt, dt[-1])

# 2. Integrate acceleration -> velocity
v_true = np.cumsum(a_true * dt)
v_noisy = np.cumsum(a_noisy * dt)

# 3. Integrate velocity -> distance

d_true = np.cumsum(v_true * dt)
d_noisy = np.cumsum(v_noisy * dt)

# 4. Plot acceleration vs noisy acceleration
plt.figure(figsize=(9, 4))
plt.plot(t, a_true, label="Actual acceleration")
plt.plot(t, a_noisy, label="Noisy acceleration", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Actual vs Noisy Acceleration")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Plot speeds 

plt.figure(figsize=(9, 4))
plt.plot(t, v_true, label="Speed from actual accel")
plt.plot(t, v_noisy, label="Speed from noisy accel", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Speed Over Time")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 4))
plt.plot(t, d_true, label="Distance from actual accel")
plt.plot(t, d_noisy, label="Distance from noisy accel", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.title("Distance Over Time")
plt.legend()
plt.tight_layout()
plt.show()


final_true = d_true[-1]
final_noisy = d_noisy[-1]
difference = abs(final_true - final_noisy)

print(f"Final distance (actual acceleration): {final_true:.4f} m")
print(f"Final distance (noisy acceleration):  {final_noisy:.4f} m")
print(f"Difference between distances:         {difference:.4f} m")
