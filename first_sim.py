import numpy as np #contains math libraries for grid/arrays and matrices
import matplotlib.pyplot as plt
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
#Step 1: Grid - Define the computational grid/simulated space
#A 2cm x 2cm square, divide into 200 x 200 gridpoints (100 micrometers apart)
Nx, Ny = 200, 200  # number of grid points in the x and y directions
dx, dy = 1e-4, 1e-4  # grid point spacing in the x and y directions [m]
kgrid = kWaveGrid([Nx, Ny], [dx, dy]) 
#N = number of grid points along the axis, dx = spacing between grid points in each dimension

#Step 2: Medium properties - The number of barriers the wave travels through 
#Simple, medium = water, it's homogenous, and has a speed of sound of 1500 m/s
medium = kWaveMedium(
    sound_speed=1500, # [m/s], water
    density=1000 # [kg/m^3], water
)

# Automatically calculate time step and number of steps needed
kgrid.makeTime(medium.sound_speed)

#Count the number of seconds it takes for the wave to travel through the medium, and how many steps it takes to simulate that time
print("dt (seconds per step):", kgrid.dt)
print("Total time simulated (seconds):", kgrid.dt * kgrid.Nt)

#Step 3: Source - how and from where the wave is made
# a small disc of high initial pressure in the center of the grid
# like dropping a pebble in a pond, the wave will propagate outwards from the center

source_p0 = np.zeros((Nx, Ny))
cx, cy = Nx // 2, Ny // 2  # center of the grid
radius = 5  # radius of the disc in grid points
for i in range(Nx):
    for j in range(Ny):
        if (i - cx) ** 2 + (j - cy) ** 2 <= radius ** 2:
            source_p0[i, j] = 1.0  # initial pressure of 1 Pa; 
            # for every point that is within the radius of the disc, set the initial pressure to 1 Pa

#Step 4: Sensor - where the wave is measured
# record the pressure at every grid point in the domain
# useful to see how the full wave moves through the medium, but practice you would record at a few end points
# it gets expensive to record at every point, and you don't need to know the pressure everywhere
sensor_mask = np.ones((Nx, Ny), dtype = bool)  # record pressure at all grid points 
sensor = kSensor(mask=sensor_mask, record='p')  # record pressure at all grid

#Step 5: Run the simulation (backend='python' since C++ binary isn't compatible with this Mac)
source = kSource()
source.p0 = source_p0
sensor_data = kspaceFirstOrder(
    kgrid, medium, source, sensor, 
    pml_inside=False, quiet=True, backend='python')
    #PML (Perfectly Matched Layer) is a boundary condition that absorbs outgoing waves to prevent reflections from the edges of the grid.
    #without it, the wave would bounce back one it hit the 200x200 grid boundary, and you would see a reflection of the wave in the sensor data; which wouldn't be realistic for a wave propagating in an infinite medium like water
    #pml_inside=False means that the PML is not applied inside the grid, only at the boundaries. This is appropriate for this simulation since we want to observe the wave propagation in the medium without any artificial reflections from the edges.
    #quiet=True suppresses the output of the simulation, so you won't see any progress messages or warnings in the console. This is useful for keeping the output clean, especially when running multiple simulations or when you don't need to monitor the simulation progress.

print("sensor_data['p'] shape:", sensor_data['p'].shape)
print("sensor_mask sum (points marked true):", sensor_mask.sum())

#Step 6: Visualize the results
# plot the initial pressure distribution
final_pressure = sensor_data['p'][:,-1].reshape(Nx, Ny)  # final pressure distribution at the last time step
plt.imshow(final_pressure, cmap='viridis') #cmap = 'viridis' is a color map that goes from dark blue to yellow, which is good for visualizing data with a wide range of values other color maps include 'plasma', 'inferno', 'magma', 'cividis', 'hot',
plt.colorbar(label='Pressure [Pa]')
plt.title('Final Pressure Distribution')
plt.xlabel('Grid Points (x)')
plt.ylabel('Grid Points (y)')
plt.show()