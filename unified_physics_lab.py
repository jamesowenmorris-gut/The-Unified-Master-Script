import numpy as np
import matplotlib.pyplot as plt

class UnifiedPhysicsLab:
    def __init__(self, size=64, dim=2):
        """
        Initializes the Physics Lab.
        :param size: Dimension size (e.g., 64 means 64x64 or 64x64x64)
        :param dim: 2 or 3 (Dimensionality of the simulation)
        """
        self.size = size
        self.dim = dim
        shape = (size,) * dim
        self.grid = np.zeros(shape, dtype=complex)
        self.velocity_field = np.zeros(shape, dtype=complex)

    def apply_physics(self, dt=0.1, friction=0.005, noise_level=0.0):
        """
        Universal evolution operator using a central difference Laplacian.
        Auto-adapts to 2D or 3D grids.
        """
        laplacian = np.zeros_like(self.grid)
        for i in range(self.dim):
            laplacian += np.roll(self.grid, 1, axis=i) + np.roll(self.grid, -1, axis=i)
        laplacian -= (2 * self.dim) * self.grid
        
        self.velocity_field = (self.velocity_field + laplacian * dt) * (1 - friction)
        self.grid += self.velocity_field * dt
        
        # Environmental Decoherence (Stochastic Noise)
        if noise_level > 0:
            noise = (np.random.randn(*self.grid.shape) + 1j * np.random.randn(*self.grid.shape)) * noise_level
            self.grid += noise

    def add_source(self, position, amplitude=1.0):
        """Adds a localized source to the grid."""
        self.grid[position] = amplitude

    def measure(self, position, radius=3):
        """Quantum Measurement: Collapses wave function at given position."""
        coords = np.indices(self.grid.shape)
        dist_sq = np.sum((coords - np.array(position).reshape(-1, *([1]*self.dim)))**2, axis=0)
        mask = dist_sq <= radius**2
        self.grid[mask] = 0 
        print(f"Observation Event at {position} triggered.")

    def measure_coherence_spectral(self):
        """Quantifies coherence using Spectral SNR (Fourier Domain)."""
        f_transform = np.fft.fftshift(np.fft.fftn(np.abs(self.grid)))
        power_spectrum = np.abs(f_transform)**2
        
        center = tuple(s // 2 for s in power_spectrum.shape)
        peak = power_spectrum[center]
        
        mask = np.ones_like(power_spectrum)
        idx = tuple(slice(c-2, c+3) for c in center)
        mask[idx] = 0
        avg_noise = np.mean(power_spectrum[mask == 1])
        return peak / (avg_noise + 1e-6)

    def visualize_intensity(self, slice_index=None):
        """Visualizes probability density (|psi|^2)."""
        plt.figure(figsize=(6, 5))
        if self.dim == 2:
            plt.imshow(np.abs(self.grid)**2, cmap='inferno')
        else:
            idx = slice_index if slice_index else self.size // 2
            plt.imshow(np.abs(self.grid[idx, :, :])**2, cmap='inferno')
        plt.title(f"Intensity Density (Dim={self.dim})")
        plt.colorbar()
        plt.show()

    def visualize_phase(self, slice_index=None):
        """Visualizes the Phase (Angle) of the complex wave function."""
        phase_map = np.angle(self.grid)
        plt.figure(figsize=(6, 5))
        if self.dim == 2:
            plt.imshow(phase_map, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        else:
            idx = slice_index if slice_index else self.size // 2
            plt.imshow(phase_map[idx, :, :], cmap='twilight', vmin=-np.pi, vmax=np.pi)
        plt.title("Phase Map (Complex Angle)")
        plt.colorbar(label='radians')
        plt.show()

# --- Execution Module ---
if __name__ == "__main__":
    # Example: 2D Interference Experiment
    print("Initializing 2D Lab...")
    lab = UnifiedPhysicsLab(size=64, dim=2)
    
    # Init double-slit
    lab.grid[10:54, 0] = 1.0
    lab.grid[:, 32] = 0
    lab.grid[20:23, 32] = 1j
    lab.grid[40:43, 32] = 1j
    
    # Run Evolution
    for _ in range(200): 
        lab.apply_physics(friction=0, noise_level=0.001)
    
    # Visualize Results
    lab.visualize_intensity()
    lab.visualize_phase()
    
    # Diagnostics
    snr = lab.measure_coherence_spectral()
    print(f"Final Spectral Coherence (SNR): {snr:.4f}")
