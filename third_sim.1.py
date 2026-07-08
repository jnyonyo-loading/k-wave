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
#note: the grid points are all positive where (0,0) is the bottom left corner of the grid!
dx, dy = 1e-4, 1e-4  # grid point spacing in the x and y directions [m]
kgrid = kWaveGrid([Nx, Ny], [dx, dy]) 
#N = number of grid points along the axis, dx = spacing between grid points in each dimension

#Step 2: Medium properties - The number of barriers the wave travels through 
# now heterogenous: water --> skull --> brain, with different sound speeds and densities for each layer
# in first and second sim, the medium was homogenous (water), so a single value could be used for sound speeed and density, but now we have to define a 2D array for each property, with different values for each layer
# meaning that every gridpoint could represent 1 value
# but because we are using a heterogenous medium, each new medium requieres a new value for sound speed and density at different gridpoints
# so a single (scalar) value is not possible - it cannot contain location information,
# what is needed is a 2D array (matrix) that can hold soundspeed, density, AND location (x,y) to define a new medium at a specific location in the grid
# so we create an empty array with the same shape as the grid, then fill in values for each medium

#Water medium values, it covers the whole grid, first.
sound_speed = np.full((Nx, Ny), 1500.0)  # water/brain, m/s
density     = np.full((Nx, Ny), 1000.0)  # water/brain, kg/m^3

# Now overwrite a region with skull values - skull slab
# skull slab - ~7mm thick (70 grid points at 100 micron spacing), positioned partway across
skull_start, skull_end = 90, 160  # x-index range
# outer table ~28%, diploe ~47%, inner table ~25% (Eisová et al. 2016, parietal bone)
outer_end  = skull_start + round(70 * 0.28)   # ~90 to 110 (2.0mm) #28% of the distance from skull start to skull end (70)
diploe_end = outer_end + round(70 * 0.47)  # ~110 to 143 (3.3mm) #47% of the distance from skull start to skull end (70)
inner_end  = skull_end  # remainder to skull_end (~143 to 160, 1.7mm) #25% of the skull start to end distance.
# could be written as inner_end = diploe_end + round(70* 0.25) - though may round to 161 instead of 160!

sound_speed[skull_start:outer_end, :]        = 2900.0  # outer table (cortical)
density[skull_start:outer_end, :]            = 1875.0

sound_speed[outer_end:diploe_end, :] = 2350.0  # diploe (spongy)
density[outer_end:diploe_end, :]     = 1600.0

sound_speed[diploe_end:inner_end, :] = 2900.0  # inner table (cortical)
density[diploe_end:inner_end, :]     = 1875.0

# that is to say: between grid points previous and final point; this is the speed, this is the density
# ':' means all y-values, so the skull layer is a vertical slab across the entire y-axis of the grid

# brain tissue beyond the skull
sound_speed[skull_end:, :] = 1550.0  # brain, m/s
density[skull_end:, :]     = 1045.0  # brain, kg/m^3
# that is to say: after 160 grid points, the rest of the grid is brain tissue
# all in all the full x-axis layout ends up as:

# x = 0 to 89 → water (background value, untouched)
# x = 90 to 159 → skull (2800 m/s / 1850 kg/m³)
# x = 160 to 199 → brain (1550 m/s / 1045 kg/m³)


medium = kWaveMedium(
    sound_speed=sound_speed,
    density=density
)

# Automatically calculate time step and number of steps needed
kgrid.makeTime(medium.sound_speed)   # uses CFL=0.3 by default, same as 'auto' would


#Count the number of seconds it takes for the wave to travel through the medium, and how many steps it takes to simulate that time
print("dt (seconds per step):", kgrid.dt)
print("Total time simulated (seconds):", kgrid.dt * kgrid.Nt)

#Step 3: Source - how and from where the wave is made
# a small disc of high initial pressure in the center of the grid
# now a disc will be continuous  ongoing sin wave from the left, before the skull

source_p_mask = np.zeros((Nx, Ny))  # blank sheet, all gridpoints marked as "not part of the source" - like a stencil
#becuase time varying pulses occur in more than one gridpoint and time step we now need to split the source into two parts: a spatial mask (location of the source) and time varying signal (the actual pressure values at each time step)
#hence a new variable source.p_mask is created to define the spatial location of the source, and source.p0 is used to define the time varying signal at the locations.
radius = 5 # radius of the disc in grid points
cx, cy = 40, Ny // 2   # off-center, well before the skull at x=90
xx, yy = np.meshgrid(np.arange(200), np.arange(200), indexing='ij') # indexing='ij' means that the first dimension of the grid corresponds to the x-axis and the second dimension corresponds to the y-axis, which is consistent with how we defined the grid earlier. This creates a grid of x and y coordinates that we can use to define the source location.
# so we use a vectorised approach instead - this creates a grid of x and y coordinates, and then we can use a single line of code to check which points are within the source radius and set those points to 1 (True) in the source_p_mask array
# xx, yy standard nomenclature is used to refer to x and y coordinates of the grid, respectively. could use x and y too.
# vectorised approach because meshgrid is taking 1D vector(x's) and combines it with 1D vector(y's) to create a 2D grid (array) of x and y coordinates. 
# whereas the nested for loop would have to go through every single grid point, check for x coordinate, then check for y coordinate, and then check if that coordinate was within the radius of the source.
# now when we fill in the mask, the meshgrid compares its empty array with the values we have noted and creates the pressure source
# like pouring paint into/over a stencil.
source_p_mask = ((xx - cx)**2 + (yy - cy)**2 <= radius**2).astype(float)
# builds the whole disc mask in one vectorised comparison instead of looping point-by-point

drive_freq = 500e3   # 500 kHz - realistic FUS territory, temporal frequency of the wave, how many times the wave oscillates per second
amplitude = 1.0       # Pa - amplitude the strength of the wave, how high the wave peaks are, how much pressure is applied to the medium

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
print("Pressure just before skull:", final_pressure[89, 100])
print("Pressure just after skull:", final_pressure[161, 100])


#Step 6: Visualize the results, with skull boundaries marked in cyan dashed lines
# plot the initial pressure distribution
final_pressure = sensor_data['p'][:,-1].reshape(Nx, Ny)  # final pressure distribution at the last time step
plt.imshow(np.abs(final_pressure).T, cmap='hot') #.T needed, to transpose the array so that the x and y axes are correctly oriented in the plot
#.imshow() places an arrays first axis from top to bottom (Y), and the second axis from left to right; 
# because we have Nx,NY in our axes, it places our x values along the y axis, and our y values along the x axis; so .T swaps the two axes that.
plt.axvline(skull_start, color='cyan', linestyle='--', linewidth=1)
plt.axvline(skull_end, color='cyan', linestyle='--', linewidth=1)
plt.axvline(outer_end, color='green', linestyle='--', linewidth=1)
plt.axvline(diploe_end, color='green', linestyle='--', linewidth=1)
plt.colorbar(label='Pressure [Pa]')
plt.title('Final Pressure Distribution')
plt.xlabel('Grid Points (x)')
plt.ylabel('Grid Points (y)')
plt.show()