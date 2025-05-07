# Example code for generating toy MD simulation data for silica
import numpy as np
import matplotlib.pyplot as plt
from openmm import *
from openmm.app import *
from openmm.unit import *

# Function to create a simple silica system
def create_silica_system(size=3, a=4.0):
    """
    Create a simple cubic silica crystal structure
    size: number of unit cells in each dimension
    a: lattice parameter in Angstroms
    """
    # Create an empty system
    system = System()
    
    # Add particles (Si and O atoms in a simple pattern)
    # In reality, silica has a more complex structure, but this is a toy model
    atoms = []
    positions = []
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                # Silicon atom at lattice points
                si_pos = [i*a, j*a, k*a]
                atoms.append('Si')
                positions.append(si_pos)
                
                # Oxygen atoms at Si-O-Si bridges (simplified)
                o_pos1 = [i*a + a/2, j*a, k*a]
                o_pos2 = [i*a, j*a + a/2, k*a]
                o_pos3 = [i*a, j*a, k*a + a/2]
                
                atoms.extend(['O', 'O', 'O'])
                positions.extend([o_pos1, o_pos2, o_pos3])
    
    # Convert to numpy array
    positions = np.array(positions) * nanometers
    
    # Create a simple force field (toy model)
    # In a real simulation, you would use a proper force field for silica
    nonbond_force = NonbondedForce()    # Nonbonded force is how atpms attract/repel at a distance
    harmonic_bond = HarmonicBondForce() # spring-like connections between atoms
    
    # Add particles to system and forces
    for i, atom_type in enumerate(atoms):
        if atom_type == 'Si':
            mass = 28.0855
            charge = 0.5  # Simplified charge
            sigma = 0.1  # Simplified Lennard-Jones parameter
            epsilon = 0.1  # Simplified Lennard-Jones parameter
        else:  # Oxygen
            mass = 15.999
            charge = -0.25  # Simplified charge
            sigma = 0.1
            epsilon = 0.1
            
        system.addParticle(mass * amu)
        nonbond_force.addParticle(charge, sigma * nanometers, epsilon * kilojoules_per_mole)
    
    # Add simple Si-O bonds (in a real model these would be more complex)
    for i in range(0, len(atoms), 4):
        if i+3 < len(atoms):
            # Bond Si to its three oxygen atoms
            harmonic_bond.addBond(i, i+1, 0.16 * nanometers, 500.0 * kilojoules_per_mole / (nanometers*nanometers))
            harmonic_bond.addBond(i, i+2, 0.16 * nanometers, 500.0 * kilojoules_per_mole / (nanometers*nanometers))
            harmonic_bond.addBond(i, i+3, 0.16 * nanometers, 500.0 * kilojoules_per_mole / (nanometers*nanometers))
    
    # Add forces to the system
    system.addForce(nonbond_force)
    system.addForce(harmonic_bond)
    
    return system, positions, atoms

# Create system
system, positions, atoms = create_silica_system(size=3)

# Set up integrator
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

# Create simulation
topology = Topology()
simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

# Minimize energy to get a reasonable starting configuration
simulation.minimizeEnergy()

# Run simulation and collect data
n_steps = 1000
data = []

for i in range(n_steps):
    if i % 100 == 0:
        print(f"Step {i}/{n_steps}")
    
    simulation.step(10)  # Take 10 steps at a time
    
    # CORRECTED: Use the proper parameters for getState()
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    
    positions = state.getPositions(asNumpy=True)
    potential_energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    kinetic_energy = state.getKineticEnergy().value_in_unit(kilojoules_per_mole)
    
    # Store data for ML training
    data.append({
        'positions': positions.value_in_unit(nanometers),
        'potential_energy': potential_energy,
        'kinetic_energy': kinetic_energy,
        'total_energy': potential_energy + kinetic_energy
    })

# Convert to numpy arrays for ML training
positions_array = np.array([d['positions'] for d in data])
energies_array = np.array([d['potential_energy'] for d in data])

print(f"Generated {n_steps} frames of MD simulation data")
print(f"Position data shape: {positions_array.shape}")
print(f"Energy data shape: {energies_array.shape}")

# Save data for ML training
np.save('silica_positions.npy', positions_array)
np.save('silica_energies.npy', energies_array)

# Visualize energy trajectory
plt.figure(figsize=(10, 6))
plt.plot([d['potential_energy'] for d in data], label='Potential Energy')
plt.plot([d['kinetic_energy'] for d in data], label='Kinetic Energy')
plt.plot([d['total_energy'] for d in data], label='Total Energy')
plt.xlabel('Frame')
plt.ylabel('Energy (kJ/mol)')
plt.legend()
plt.title('Energy Trajectory')
plt.savefig('energy_trajectory.png')
plt.show()

# Visualize a slice of the positions data to show what it looks like
plt.figure(figsize=(15, 10))

# Plot 3D positions for the first frame
frame_idx = 0
positions_frame = positions_array[frame_idx]

# Separate Si and O atoms based on our system creation pattern
# Every 4th atom is Si, the rest are O
si_indices = np.arange(0, len(positions_frame), 4)
o_indices = np.array([i for i in range(len(positions_frame)) if i not in si_indices])

# 3D scatter plot
ax = plt.subplot(2, 2, 1, projection='3d')
ax.scatter(positions_frame[si_indices, 0], positions_frame[si_indices, 1], positions_frame[si_indices, 2], 
           c='blue', label='Si', s=100, alpha=0.7)
ax.scatter(positions_frame[o_indices, 0], positions_frame[o_indices, 1], positions_frame[o_indices, 2], 
           c='red', label='O', s=50, alpha=0.5)
ax.set_title(f'3D Structure (Frame {frame_idx})')
ax.set_xlabel('X (nm)')
ax.set_ylabel('Y (nm)')
ax.set_zlabel('Z (nm)')
ax.legend()

# Plot X, Y, Z position distributions
ax2 = plt.subplot(2, 2, 2)
ax2.hist(positions_frame[:, 0], bins=20, alpha=0.5, label='X')
ax2.hist(positions_frame[:, 1], bins=20, alpha=0.5, label='Y')
ax2.hist(positions_frame[:, 2], bins=20, alpha=0.5, label='Z')
ax2.set_title('Position Distributions')
ax2.set_xlabel('Position (nm)')
ax2.set_ylabel('Count')
ax2.legend()

# Plot position changes over time for a single atom
atom_idx = 0  # First atom
ax3 = plt.subplot(2, 2, 3)
ax3.plot(np.arange(n_steps), positions_array[:, atom_idx, 0], label='X')
ax3.plot(np.arange(n_steps), positions_array[:, atom_idx, 1], label='Y')
ax3.plot(np.arange(n_steps), positions_array[:, atom_idx, 2], label='Z')
ax3.set_title(f'Position of Atom {atom_idx} Over Time')
ax3.set_xlabel('Frame')
ax3.set_ylabel('Position (nm)')
ax3.legend()

# Plot displacement between frames
displacements = np.sqrt(np.sum(np.diff(positions_array, axis=0)**2, axis=2))
ax4 = plt.subplot(2, 2, 4)
ax4.hist(displacements.flatten(), bins=30)
ax4.set_title('Atom Displacements Between Frames')
ax4.set_xlabel('Displacement (nm)')
ax4.set_ylabel('Count')

plt.tight_layout()
plt.savefig('md_data_visualization.png')
plt.show()

print("Simulation and visualization complete!")