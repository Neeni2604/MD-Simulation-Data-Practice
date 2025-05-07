import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Load the data
positions = np.load('silica_positions.npy')
energies = np.load('silica_energies.npy')

print(f"We have {positions.shape[0]} frames of simulation")
print(f"Each frame has {positions.shape[1]} atoms")
print(f"Position shape: {positions.shape}")

# Make a simple animation of atom movements to understand the dynamics
# Focus on just the first 100 frames to keep it manageable
frames_to_show = 100
skip_atoms = 4  # Only show every 4th atom to reduce clutter

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Every 4th atom is Si, the rest are O (based on our system creation)
atom_indices = range(0, positions.shape[1], skip_atoms)
si_indices = [i for i in atom_indices if i % 4 == 0]
o_indices = [i for i in atom_indices if i % 4 != 0]

# Set up scatter plots for Si and O atoms
si_scatter = ax.scatter([], [], [], c='blue', marker='o', s=100, label='Si', alpha=0.7)
o_scatter = ax.scatter([], [], [], c='red', marker='o', s=50, label='O', alpha=0.5)

# Set plot limits
max_range = np.max(positions[:frames_to_show, :, :].max(axis=(0,1)) - 
                   positions[:frames_to_show, :, :].min(axis=(0,1))) / 2
mid_x = (positions[:frames_to_show, :, 0].max() + positions[:frames_to_show, :, 0].min()) / 2
mid_y = (positions[:frames_to_show, :, 1].max() + positions[:frames_to_show, :, 1].min()) / 2
mid_z = (positions[:frames_to_show, :, 2].max() + positions[:frames_to_show, :, 2].min()) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('X (nm)')
ax.set_ylabel('Y (nm)')
ax.set_zlabel('Z (nm)')
ax.set_title('Silica MD Simulation')
ax.legend()

# Add energy plot
energy_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

# Animation update function
def update(frame):
    # Update positions
    si_scatter._offsets3d = (positions[frame, si_indices, 0], 
                             positions[frame, si_indices, 1], 
                             positions[frame, si_indices, 2])
    
    o_scatter._offsets3d = (positions[frame, o_indices, 0], 
                           positions[frame, o_indices, 1], 
                           positions[frame, o_indices, 2])
    
    # Update energy text
    energy_text.set_text(f'Frame: {frame}, Energy: {energies[frame]:.2f} kJ/mol')
    
    return si_scatter, o_scatter, energy_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frames_to_show, 
                              interval=100, blit=False)

# Save animation
ani.save('silica_animation.gif', writer='pillow', fps=10)
plt.close()

print("Animation saved as 'silica_animation.gif'")

# Also create a static plot for easier viewing
plt.figure(figsize=(16, 8))

# Plot energy over time
plt.subplot(1, 2, 1)
plt.plot(energies[:100])
plt.xlabel('Frame')
plt.ylabel('Potential Energy (kJ/mol)')
plt.title('Energy vs. Time')

# Plot the final structure - FIX HERE
ax3d = plt.subplot(1, 2, 2, projection='3d')
frame = 99  # Last frame in our animation
 
# Plot Si atoms - Fixed the scatter function
ax3d.scatter(positions[frame, si_indices, 0], 
             positions[frame, si_indices, 1], 
             positions[frame, si_indices, 2],
             c='blue', marker='o', s=100, label='Si', alpha=0.7)

# Plot O atoms - Fixed the scatter function
ax3d.scatter(positions[frame, o_indices, 0], 
             positions[frame, o_indices, 1], 
             positions[frame, o_indices, 2],
             c='red', marker='o', s=50, label='O', alpha=0.5)

ax3d.set_xlabel('X (nm)')
ax3d.set_ylabel('Y (nm)')
ax3d.set_zlabel('Z (nm)')
ax3d.set_title('Final Structure (Frame 99)')
ax3d.legend()

plt.tight_layout()
plt.savefig('silica_md_summary.png')
plt.show()

print("Summary plot saved as 'silica_md_summary.png'")