"""
SPECT Detector Calibration and Correction Pipeline
Simulates quality assurance procedures and calibration algorithms for gamma camera systems

Author: Abraham Taye
Purpose: Demonstrate detector calibration expertise for Siemens Healthineers position
Reference: NEMA Standards Publication NU 1-2018
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import griddata
import json
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class DetectorConfig:
    """Configuration for gamma camera detector"""
    crystal_size: Tuple[int, int] = (64, 64)  # pixels
    pixel_size: float = 4.0  # mm
    num_pmts: int = 4  # Number of PMTs
    pmt_layout: str = "square"  # "square" or "hexagonal"
    crystal_thickness: float = 9.5  # mm (NaI(Tl))
    energy_window: Tuple[float, float] = (126, 154)  # keV (20% window at 140 keV)

class FloodFieldGenerator:
    """Generate synthetic flood field data with realistic non-uniformities"""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.x_size, self.y_size = config.crystal_size
        
    def generate_flood_field(self, 
                            mean_counts: int = 1000,
                            add_pmt_variations: bool = True,
                            add_edge_effects: bool = True,
                            add_defects: bool = True) -> np.ndarray:
        """
        Generate synthetic flood field image with various non-uniformities
        
        Args:
            mean_counts: Average counts per pixel
            add_pmt_variations: Include PMT gain variations
            add_edge_effects: Include edge packing effects
            add_defects: Include random crystal defects
            
        Returns:
            2D array of count data
        """
        flood_field = np.ones(self.config.crystal_size) * mean_counts
        
        # Add PMT gain variations (quadrant-based)
        if add_pmt_variations:
            flood_field = self._add_pmt_variations(flood_field)
        
        # Add edge packing effects
        if add_edge_effects:
            flood_field = self._add_edge_effects(flood_field)
        
        # Add crystal defects
        if add_defects:
            flood_field = self._add_crystal_defects(flood_field)
        
        # Add Poisson noise
        flood_field = np.random.poisson(flood_field)
        
        return flood_field.astype(float)
    
    def _add_pmt_variations(self, data: np.ndarray) -> np.ndarray:
        """Simulate PMT gain variations"""
        x_mid = self.x_size // 2
        y_mid = self.y_size // 2
        
        # Define gain for each PMT quadrant
        gains = {
            'top_left': 1.0,
            'top_right': 0.85,
            'bottom_left': 0.92,
            'bottom_right': 0.88
        }
        
        result = data.copy()
        result[:y_mid, :x_mid] *= gains['top_left']
        result[:y_mid, x_mid:] *= gains['top_right']
        result[y_mid:, :x_mid] *= gains['bottom_left']
        result[y_mid:, x_mid:] *= gains['bottom_right']
        
        # Add smooth spatial variations
        x, y = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size))
        spatial_variation = 0.05 * np.sin(2 * np.pi * x / self.x_size) * \
                           np.cos(2 * np.pi * y / self.y_size)
        result *= (1 + spatial_variation)
        
        return result
    
    def _add_edge_effects(self, data: np.ndarray) -> np.ndarray:
        """Simulate edge packing effects (light collection loss at edges)"""
        x, y = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size))
        center_x, center_y = self.x_size / 2, self.y_size / 2
        
        # Distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Edge drop-off (30% reduction at edges)
        edge_factor = 1.0 - 0.3 * (dist / max_dist)**2
        
        return data * edge_factor
    
    def _add_crystal_defects(self, data: np.ndarray) -> np.ndarray:
        """Add random crystal defects (dead spots, cracks)"""
        result = data.copy()
        
        # Random dead pixels (2%)
        num_dead = int(0.02 * self.x_size * self.y_size)
        dead_x = np.random.randint(0, self.x_size, num_dead)
        dead_y = np.random.randint(0, self.y_size, num_dead)
        result[dead_y, dead_x] *= 0.3
        
        # Simulated crack (linear defect)
        if np.random.random() < 0.3:  # 30% chance of crack
            crack_y = np.random.randint(10, self.y_size - 10)
            crack_width = 2
            result[crack_y:crack_y+crack_width, :] *= 0.6
        
        return result

class DetectorCalibration:
    """Main calibration and correction pipeline"""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.correction_map = None
        self.energy_correction_map = None
        self.linearity_correction = None
        
    def generate_uniformity_correction_map(self, flood_field: np.ndarray) -> np.ndarray:
        """
        Generate uniformity correction map from flood field acquisition
        
        Args:
            flood_field: Raw flood field data
            
        Returns:
            Correction map (multiplicative factors)
        """
        # Apply smoothing to reduce noise
        smoothed = gaussian_filter(flood_field, sigma=2.0)
        
        # Calculate mean
        mean_value = np.mean(smoothed)
        
        # Generate correction factors
        correction_map = mean_value / (smoothed + 1e-6)  # Avoid division by zero
        
        # Cap correction factors to reasonable range
        correction_map = np.clip(correction_map, 0.3, 3.0)
        
        self.correction_map = correction_map
        return correction_map
    
    def apply_uniformity_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply uniformity correction to an image"""
        if self.correction_map is None:
            raise ValueError("Correction map not generated. Run generate_uniformity_correction_map first.")
        
        return image * self.correction_map
    
    def generate_energy_correction_map(self, 
                                      energy_data: np.ndarray,
                                      target_energy: float = 140.0) -> np.ndarray:
        """
        Generate position-dependent energy correction
        
        Args:
            energy_data: 2D array of measured photopeak energies
            target_energy: Target photopeak energy (keV)
            
        Returns:
            Energy correction map
        """
        # Smooth energy map
        smoothed_energy = gaussian_filter(energy_data, sigma=2.0)
        
        # Calculate correction factors
        energy_correction = target_energy / (smoothed_energy + 1e-6)
        
        self.energy_correction_map = energy_correction
        return energy_correction
    
    def apply_energy_correction(self, energy_values: np.ndarray) -> np.ndarray:
        """Apply position-dependent energy correction"""
        if self.energy_correction_map is None:
            raise ValueError("Energy correction map not generated.")
        
        return energy_values * self.energy_correction_map
    
    def calculate_linearity_distortion(self, 
                                      reference_grid: np.ndarray,
                                      measured_grid: np.ndarray) -> Dict:
        """
        Calculate spatial linearity distortion
        
        Args:
            reference_grid: Known positions of point sources
            measured_grid: Measured positions
            
        Returns:
            Dictionary with distortion metrics
        """
        # Calculate displacement vectors
        displacement = measured_grid - reference_grid
        displacement_magnitude = np.sqrt(np.sum(displacement**2, axis=-1))
        
        # Calculate distortion metrics
        max_distortion = np.max(displacement_magnitude)
        mean_distortion = np.mean(displacement_magnitude)
        
        self.linearity_correction = {
            'displacement_field': displacement,
            'max_distortion': max_distortion,
            'mean_distortion': mean_distortion
        }
        
        return self.linearity_correction

class PerformanceMetrics:
    """Calculate NEMA NU-1 performance metrics"""
    
    @staticmethod
    def calculate_integral_uniformity(image: np.ndarray, 
                                     use_ufov: bool = True) -> float:
        """
        Calculate integral uniformity (NEMA NU-1)
        
        Args:
            image: Flood field image
            use_ufov: Use Useful Field of View (UFOV) vs Central FOV (CFOV)
            
        Returns:
            Integral uniformity percentage
        """
        if use_ufov:
            # UFOV: 95% of detector area
            mask = PerformanceMetrics._get_ufov_mask(image.shape)
        else:
            # CFOV: 75% of detector area
            mask = PerformanceMetrics._get_cfov_mask(image.shape)
        
        roi = image[mask]
        max_count = np.max(roi)
        min_count = np.min(roi)
        
        integral_uniformity = ((max_count - min_count) / (max_count + min_count)) * 100
        return integral_uniformity
    
    @staticmethod
    def calculate_differential_uniformity(image: np.ndarray,
                                         use_ufov: bool = True) -> float:
        """
        Calculate differential uniformity (NEMA NU-1)
        
        Maximum difference between adjacent pixels in 5-pixel linear array
        """
        if use_ufov:
            mask = PerformanceMetrics._get_ufov_mask(image.shape)
        else:
            mask = PerformanceMetrics._get_cfov_mask(image.shape)
        
        roi = image.copy()
        roi[~mask] = 0
        
        # Horizontal differences
        diff_h = np.abs(np.diff(roi, axis=1))
        max_diff_h = np.max(diff_h)
        
        # Vertical differences
        diff_v = np.abs(np.diff(roi, axis=0))
        max_diff_v = np.max(diff_v)
        
        max_diff = max(max_diff_h, max_diff_v)
        mean_count = np.mean(roi[mask])
        
        differential_uniformity = (max_diff / mean_count) * 100
        return differential_uniformity
    
    @staticmethod
    def calculate_energy_resolution(energies: np.ndarray,
                                   photopeak: float = 140.0) -> float:
        """
        Calculate energy resolution at photopeak (FWHM %)
        
        Args:
            energies: Array of measured energies
            photopeak: Nominal photopeak energy (keV)
            
        Returns:
            Energy resolution as FWHM percentage
        """
        # Select events near photopeak (±10%)
        window = (energies > 0.9 * photopeak) & (energies < 1.1 * photopeak)
        peak_energies = energies[window]
        
        if len(peak_energies) < 100:
            return np.nan
        
        # Calculate FWHM
        mean_energy = np.mean(peak_energies)
        std_energy = np.std(peak_energies)
        fwhm = 2.355 * std_energy
        
        energy_resolution = (fwhm / mean_energy) * 100
        return energy_resolution
    
    @staticmethod
    def calculate_spatial_resolution(line_spread_function: np.ndarray) -> float:
        """
        Calculate spatial resolution from line spread function
        
        Returns:
            FWHM in mm
        """
        # Fit Gaussian and calculate FWHM
        # Simplified: use direct measurement
        max_value = np.max(line_spread_function)
        half_max = max_value / 2
        
        above_half = line_spread_function > half_max
        fwhm_pixels = np.sum(above_half)
        
        # Convert to mm (assuming pixel_size is known)
        pixel_size = 4.0  # mm
        fwhm_mm = fwhm_pixels * pixel_size
        
        return fwhm_mm
    
    @staticmethod
    def _get_ufov_mask(shape: Tuple[int, int]) -> np.ndarray:
        """Get Useful Field of View mask (95% of detector)"""
        h, w = shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        radius = min(h, w) * 0.475  # 95% diameter
        
        mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        return mask
    
    @staticmethod
    def _get_cfov_mask(shape: Tuple[int, int]) -> np.ndarray:
        """Get Central Field of View mask (75% of detector)"""
        h, w = shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        radius = min(h, w) * 0.375  # 75% diameter
        
        mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        return mask

def visualize_calibration_results(raw_flood: np.ndarray,
                                  corrected_flood: np.ndarray,
                                  correction_map: np.ndarray,
                                  metrics_raw: Dict,
                                  metrics_corrected: Dict):
    """Create comprehensive visualization of calibration results"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Raw Flood Field
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(raw_flood, cmap='hot', aspect='auto')
    ax1.set_title('Raw Flood Field (Uncorrected)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Pixel')
    ax1.set_ylabel('Y Pixel')
    plt.colorbar(im1, ax=ax1, label='Counts')
    
    # 2. Correction Map
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(correction_map, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.5)
    ax2.set_title('Uniformity Correction Map', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Pixel')
    ax2.set_ylabel('Y Pixel')
    plt.colorbar(im2, ax=ax2, label='Correction Factor')
    
    # 3. Corrected Flood Field
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(corrected_flood, cmap='hot', aspect='auto')
    ax3.set_title('Corrected Flood Field', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X Pixel')
    ax3.set_ylabel('Y Pixel')
    plt.colorbar(im3, ax=ax3, label='Counts')
    
    # 4. Line Profiles (Center Row)
    ax4 = plt.subplot(2, 3, 4)
    center_row = raw_flood.shape[0] // 2
    ax4.plot(raw_flood[center_row, :], 'r-', linewidth=2, label='Raw', alpha=0.7)
    ax4.plot(corrected_flood[center_row, :], 'b-', linewidth=2, label='Corrected', alpha=0.7)
    ax4.set_title('Central Row Profile', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X Pixel')
    ax4.set_ylabel('Counts')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Uniformity Histograms
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(raw_flood.flatten(), bins=50, alpha=0.6, color='red', label='Raw', density=True)
    ax5.hist(corrected_flood.flatten(), bins=50, alpha=0.6, color='blue', label='Corrected', density=True)
    ax5.set_title('Count Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Counts')
    ax5.set_ylabel('Probability Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    metrics_text = f"""
    PERFORMANCE METRICS COMPARISON
    {'='*50}
    
    INTEGRAL UNIFORMITY (UFOV):
      Raw:       {metrics_raw['integral_uniformity']:.2f}%
      Corrected: {metrics_corrected['integral_uniformity']:.2f}%
      Improvement: {metrics_raw['integral_uniformity'] - metrics_corrected['integral_uniformity']:.2f}%
    
    DIFFERENTIAL UNIFORMITY (UFOV):
      Raw:       {metrics_raw['differential_uniformity']:.2f}%
      Corrected: {metrics_corrected['differential_uniformity']:.2f}%
      Improvement: {metrics_raw['differential_uniformity'] - metrics_corrected['differential_uniformity']:.2f}%
    
    COEFFICIENT OF VARIATION:
      Raw:       {metrics_raw['cv']:.2f}%
      Corrected: {metrics_corrected['cv']:.2f}%
    
    NEMA ACCEPTANCE CRITERIA:
      Integral Uniformity:   < 5% (PASS/FAIL)
      Differential Uniformity: < 3% (PASS/FAIL)
    
    STATUS: {'PASS ✓' if metrics_corrected['integral_uniformity'] < 5 else 'FAIL ✗'}
    """
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('calibration_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'calibration_results.png'")
    plt.show()

def main():
    """Main calibration workflow demonstration"""
    
    print("="*70)
    print("SPECT DETECTOR CALIBRATION AND QUALITY ASSURANCE PIPELINE")
    print("="*70)
    
    # 1. Initialize system
    config = DetectorConfig()
    generator = FloodFieldGenerator(config)
    calibrator = DetectorCalibration(config)
    
    print("\n1. Generating synthetic flood field acquisition...")
    raw_flood = generator.generate_flood_field(
        mean_counts=10000,
        add_pmt_variations=True,
        add_edge_effects=True,
        add_defects=True
    )
    print(f"   Flood field shape: {raw_flood.shape}")
    print(f"   Total counts: {np.sum(raw_flood):,.0f}")
    
    # 2. Calculate raw metrics
    print("\n2. Calculating raw performance metrics...")
    metrics_raw = {
        'integral_uniformity': PerformanceMetrics.calculate_integral_uniformity(raw_flood),
        'differential_uniformity': PerformanceMetrics.calculate_differential_uniformity(raw_flood),
        'cv': (np.std(raw_flood) / np.mean(raw_flood)) * 100
    }
    
    print(f"   Integral Uniformity (UFOV): {metrics_raw['integral_uniformity']:.2f}%")
    print(f"   Differential Uniformity: {metrics_raw['differential_uniformity']:.2f}%")
    print(f"   Coefficient of Variation: {metrics_raw['cv']:.2f}%")
    
    # 3. Generate correction map
    print("\n3. Generating uniformity correction map...")
    correction_map = calibrator.generate_uniformity_correction_map(raw_flood)
    print(f"   Correction factors range: {np.min(correction_map):.2f} to {np.max(correction_map):.2f}")
    
    # 4. Apply corrections
    print("\n4. Applying uniformity corrections...")
    corrected_flood = calibrator.apply_uniformity_correction(raw_flood)
    
    # 5. Calculate corrected metrics
    print("\n5. Calculating corrected performance metrics...")
    metrics_corrected = {
        'integral_uniformity': PerformanceMetrics.calculate_integral_uniformity(corrected_flood),
        'differential_uniformity': PerformanceMetrics.calculate_differential_uniformity(corrected_flood),
        'cv': (np.std(corrected_flood) / np.mean(corrected_flood)) * 100
    }
    
    print(f"   Integral Uniformity (UFOV): {metrics_corrected['integral_uniformity']:.2f}%")
    print(f"   Differential Uniformity: {metrics_corrected['differential_uniformity']:.2f}%")
    print(f"   Coefficient of Variation: {metrics_corrected['cv']:.2f}%")
    
    # 6. Evaluate against NEMA standards
    print("\n6. Evaluating against NEMA NU-1 acceptance criteria...")
    nema_pass = (metrics_corrected['integral_uniformity'] < 5.0 and 
                metrics_corrected['differential_uniformity'] < 3.0)
    
    print(f"   NEMA Integral Uniformity < 5%: {'PASS ✓' if metrics_corrected['integral_uniformity'] < 5 else 'FAIL ✗'}")
    print(f"   NEMA Differential Uniformity < 3%: {'PASS ✓' if metrics_corrected['differential_uniformity'] < 3 else 'FAIL ✗'}")
    print(f"\n   Overall Status: {'SYSTEM QUALIFIED ✓' if nema_pass else 'REQUIRES RECALIBRATION ✗'}")
    
    # 7. Visualize results
    print("\n7. Generating visualization...")
    visualize_calibration_results(raw_flood, corrected_flood, correction_map,
                                 metrics_raw, metrics_corrected)
    
    # 8. Save results
    print("\n8. Saving calibration data...")
    results = {
        'configuration': {
            'crystal_size': config.crystal_size,
            'pixel_size': config.pixel_size,
            'energy_window': config.energy_window
        },
        'metrics_raw': metrics_raw,
        'metrics_corrected': metrics_corrected,
        'improvement': {
            'integral_uniformity': metrics_raw['integral_uniformity'] - metrics_corrected['integral_uniformity'],
            'differential_uniformity': metrics_raw['differential_uniformity'] - metrics_corrected['differential_uniformity']
        },
        'nema_compliance': nema_pass
    }
    
    with open('calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save correction map
    np.save('correction_map.npy', correction_map)
    
    print("   Results saved:")
    print("     - calibration_results.json")
    print("     - correction_map.npy")
    print("     - calibration_results.png")
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()