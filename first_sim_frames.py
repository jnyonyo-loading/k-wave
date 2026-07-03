import numpy as np
import matplotlib.pyplot as plt
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder

# Step 1: Grid
Nx, Ny = 200, 200
dx, dy = 1e-4, 1e-4
kgrid = kWaveGrid([Nx, Ny], [dx, dy])

# Step 2: Medium
medium = kWaveMedium(sound_speed=1500, density=1000)
kgrid.makeTime(medium.sound_speed)

print("dt (seconds per step):", kgrid.dt)
print("Total time simulated (seconds):", kgrid.dt * kgrid.Nt)

# Step 3: Source
source_p0 = np.zeros((Nx, Ny))
cx, cy = Nx // 2, Ny // 2
radius = 5
for i in range(Nx):
    for j in range(Ny):
        if (i - cx) ** 2 + (j - cy) ** 2 <= radius ** 2:
            source_p0[i, j] = 1.0

source = kSource()
source.p0 = source_p0

# Step 4: Sensor
sensor_mask = np.ones((Nx, Ny), dtype=bool)
sensor = kSensor(mask=sensor_mask, record=['p'])

# Step 5: Run
sensor_data = kspaceFirstOrder(
    kgrid, medium, source, sensor,
    pml_inside=False, quiet=True, backend='python'
)

# Step 6: Plot several frames across time, side by side
num_frames = 6
total_steps = sensor_data['p'].shape[1]
frame_indices = np.linspace(0, total_steps - 1, num_frames, dtype=int)

fig, axes = plt.subplots(1, num_frames, figsize=(18, 3))
for ax, idx in zip(axes, frame_indices):
    frame = sensor_data['p'][:, idx].reshape(Nx, Ny)
    im = ax.imshow(frame, cmap='viridis')
    ax.set_title(f"step {idx}")
    ax.axis('off')

fig.colorbar(im, ax=axes, label='Pressure [Pa]', shrink=0.8)
plt.suptitle('Wave Propagation Over Time')
plt.show()