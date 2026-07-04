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
kgrid.makeTime(medium.sound_speed)   # uses CFL=0.3 by default, same as 'auto' would


#Count the number of seconds it takes for the wave to travel through the medium, and how many steps it takes to simulate that time
print("dt (seconds per step):", kgrid.dt)
print("Total time simulated (seconds):", kgrid.dt * kgrid.Nt)

#Step 3: Source - how and from where the wave is made
# a small disc of high initial pressure in the center of the grid
# now a disc will be continuous, producing an ongoing sin wave, rather than a single pulse
# like holding a vibrating rod in the pond instead of dropping a pebble once
# --- Time array (needed now, since we have to build a signal against it) ---

source_p_mask = np.zeros((Nx, Ny))  # blank sheet, all gridpoints marked as "not part of the source"
#source_p0 not needed anymore, since we are not using a single pulse, but a continuous wave
#becuase time varying pulses occur in more than one gridpoint and time step we now need to split the source into two parts: a spatial mask (location of the source) and time varying signal (the actual pressure values at each time step)
#previously we had one single source pressure at one position and one time step, so source_p0 could be used for both the spatial and temporal parts of the source.
#hence a new variable source.p_mask is created to define the spatial location of the source, and source.p0 is used to define the time varying signal at that location.
radius = 5 # radius of the disc in grid points
cx, cy = Nx // 2, Ny // 2  # center of the grid
xx, yy = np.meshgrid(np.arange(200), np.arange(200), indexing='ij') # indexing='ij' means that the first dimension of the grid corresponds to the x-axis and the second dimension corresponds to the y-axis, which is consistent with how we defined the grid earlier. This creates a grid of x and y coordinates that we can use to define the source location.
## nested for loop could work here but is slow and inefficient, the grid gets bigger, you dont want python having to loop through every single grid point to check if it's in the source or not, 
# so we use a vectorized approach instead - this creates a grid of x and y coordinates, and then we can use a single line of code to check which points are within the source radius and set those points to 1 (True) in the source_p_mask array
source_p_mask = ((xx - cx)**2 + (yy - cy)**2 <= radius**2).astype(float)
# builds the whole disc mask in one vectorized comparison instead of looping point-by-point

drive_freq = 500e3   # 500 kHz - realistic FUS territory, well within the grid's frequency limit
amplitude = 1.0       # keeping this in arbitrary units for now, same as the p0 amplitude

source_p = amplitude * np.sin(2 * np.pi * drive_freq * kgrid.t_array)
# a single time series - the same signal gets applied to every point marked in source_p_mask


#Step 4: Sensor - where the wave is measured
# record the pressure at every grid point in the domain
# useful to see how the full wave moves through the medium, but practice you would record at a few end points
# it gets expensive to record at every point, and you don't need to know the pressure everywhere
sensor_mask = np.ones((Nx, Ny), dtype = bool)  # record pressure at all grid points. ones set every point to True, so the sensor will record at every grid point
sensor = kSensor(mask=sensor_mask, record='p')  # record pressure at all grid
#mask parameter is grid of yes/no (boolean = true/false) values, where True means record pressure at that point, and False means don't record pressure at that point
#'p' means record pressure (quantity), other options include 'u' for particle velocity, 'I' for intensity, etc.

#Step 5: Run the simulation (backend='python' since C++ binary isn't compatible with this Mac)
source = kSource() #empty source object, will be filled with initial pressure distribution defined earlier/below
source.p_mask = source_p_mask #source.p_mask is not a new variable, it's an attributes (propert) of the source object that is being set to the spatial mask defined earlier. This is how the simulation knows where the source is located.
source.p = source_p #source.p is not a new variable, it's an attribute (property) of the source object that is being set to the time varying signal defined earlier. This is how the simulation knows what the time varying signal is, and it will use this information to calculate how the wave propagates through the medium over time.
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
plt.imshow(np.abs(final_pressure), cmap='hot')
plt.colorbar(label='Pressure [Pa]')
plt.title('Final Pressure Distribution')
plt.xlabel('Grid Points (x)')
plt.ylabel('Grid Points (y)')
plt.show()