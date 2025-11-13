"""
Advanced Quality Assurance Analysis for SPECT Detectors
Includes temporal trending, multi-isotope calibration, and automated QA reporting

Reference: NEMA NU 1-2018, ACR Gamma Camera QA Procedures
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from detector_calibration import (
    DetectorConfig, FloodFieldGenerator, 
    DetectorCalibration, PerformanceMetrics
)

class TemporalQA:
    """Track detector performance over time"""
    
    def __init__(self):
        self.qa_history = []
        
    def add_qa_measurement(self, 
                          date: datetime,
                          metrics: dict,
                          technologist: str = "Unknown"):
        """Add QA measurement to history"""
        entry = {
            'date': date,
            'integral_uniformity': metrics['integral_uniformity'],
            'differential_uniformity': metrics['differential_uniformity'],
            'technologist': technologist,
            'pass_fail': metrics.get('nema_pass', True)
        }
        self.qa_history.append(entry)
    
    def generate_trend_report(self, days_back: int = 30):
        """Generate trending analysis"""
        if not self.qa_history:
            print("No QA history available")
            return
        
        df = pd.DataFrame(self.qa_history)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to requested time period
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df_recent = df[df['date'] >= cutoff_date]
        
        if len(df_recent) == 0:
            print(f"No data in last {days_back} days")
            return
        
        # Create trend plots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Integral uniformity trend
        ax1 = axes[0]
        ax1.plot(df_recent['date'], df_recent['integral_uniformity'], 
                'o-', linewidth=2, markersize=8, color='blue', label='Measured')
        ax1.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='NEMA Limit')
        ax1.fill_between(df_recent['date'], 0, 5.0, alpha=0.2, color='green', label='Acceptable Range')
        ax1.set_ylabel('Integral Uniformity (%)', fontsize=12)
        ax1.set_title('Daily QA Trend - Integral Uniformity', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(10, df_recent['integral_uniformity'].max() * 1.2))
        
        # Differential uniformity trend
        ax2 = axes[1]
        ax2.plot(df_recent['date'], df_recent['differential_uniformity'],
                'o-', linewidth=2, markersize=8, color='orange', label='Measured')
        ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='NEMA Limit')
        ax2.fill_between(df_recent['date'], 0, 3.0, alpha=0.2, color='green', label='Acceptable Range')
        ax2.set_ylabel('Differential Uniformity (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Daily QA Trend - Differential Uniformity', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(6, df_recent['differential_uniformity'].max() * 1.2))
        
        plt.tight_layout()
        plt.savefig('qa_trend_report.png', dpi=300, bbox_inches='tight')
        print("\nTrend report saved as 'qa_trend_report.png'")
        plt.show()
        
        # Statistical summary
        print("\n" + "="*70)
        print("QA TRENDING ANALYSIS SUMMARY")
        print("="*70)
        print(f"Period: {df_recent['date'].min().strftime('%Y-%m-%d')} to {df_recent['date'].max().strftime('%Y-%m-%d')}")
        print(f"Number of QA tests: {len(df_recent)}")
        print(f"\nIntegral Uniformity:")
        print(f"  Mean: {df_recent['integral_uniformity'].mean():.2f}%")
        print(f"  Std Dev: {df_recent['integral_uniformity'].std():.2f}%")
        print(f"  Range: {df_recent['integral_uniformity'].min():.2f}% - {df_recent['integral_uniformity'].max():.2f}%")
        print(f"\nDifferential Uniformity:")
        print(f"  Mean: {df_recent['differential_uniformity'].mean():.2f}%")
        print(f"  Std Dev: {df_recent['differential_uniformity'].std():.2f}%")
        print(f"  Range: {df_recent['differential_uniformity'].min():.2f}% - {df_recent['differential_uniformity'].max():.2f}%")
        
        # Pass/Fail rate
        pass_rate = (df_recent['pass_fail'].sum() / len(df_recent)) * 100
        print(f"\nPass Rate: {pass_rate:.1f}%")
        
        # Detect trends (simple linear regression)
        from scipy import stats
        days_numeric = (df_recent['date'] - df_recent['date'].min()).dt.days
        
        slope_iu, _, _, p_value_iu, _ = stats.linregress(days_numeric, df_recent['integral_uniformity'])
        slope_du, _, _, p_value_du, _ = stats.linregress(days_numeric, df_recent['differential_uniformity'])
        
        print("\nTrend Analysis:")
        if p_value_iu < 0.05:
            trend_iu = "increasing" if slope_iu > 0 else "decreasing"
            print(f"  Integral Uniformity: Significant {trend_iu} trend detected (p={p_value_iu:.3f})")
        else:
            print(f"  Integral Uniformity: Stable (p={p_value_iu:.3f})")
            
        if p_value_du < 0.05:
            trend_du = "increasing" if slope_du > 0 else "decreasing"
            print(f"  Differential Uniformity: Significant {trend_du} trend detected (p={p_value_du:.3f})")
        else:
            print(f"  Differential Uniformity: Stable (p={p_value_du:.3f})")

class MultiIsotopeCalibration:
    """Handle calibration for multiple isotopes"""
    
    ISOTOPES = {
        'Tc-99m': {'energy': 140, 'window': (126, 154), 'color': 'blue'},
        'I-123': {'energy': 159, 'window': (143, 175), 'color': 'green'},
        'In-111': {'energy': (171, 245), 'window': ((154, 188), (220, 270)), 'color': 'orange'},
        'Tl-201': {'energy': (71, 167), 'window': ((64, 78), (150, 184)), 'color': 'purple'}
    }
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.calibrations = {}
        
    def calibrate_isotope(self, isotope_name: str):
        """Perform isotope-specific calibration"""
        if isotope_name not in self.ISOTOPES:
            raise ValueError(f"Unknown isotope: {isotope_name}")
        
        print(f"\nCalibrating for {isotope_name}...")
        isotope = self.ISOTOPES[isotope_name]
        
        # Generate flood field for this isotope
        generator = FloodFieldGenerator(self.config)
        flood = generator.generate_flood_field(mean_counts=5000)
        
        # Create calibration
        calibrator = DetectorCalibration(self.config)
        correction_map = calibrator.generate_uniformity_correction_map(flood)
        
        # Calculate metrics
        metrics = {
            'integral_uniformity': PerformanceMetrics.calculate_integral_uniformity(flood),
            'differential_uniformity': PerformanceMetrics.calculate_differential_uniformity(flood)
        }
        
        self.calibrations[isotope_name] = {
            'correction_map': correction_map,
            'metrics': metrics,
            'energy': isotope['energy']
        }
        
        print(f"  Integral Uniformity: {metrics['integral_uniformity']:.2f}%")
        print(f"  Differential Uniformity: {metrics['differential_uniformity']:.2f}%")
        
    def compare_isotopes(self):
        """Compare calibration across isotopes"""
        if len(self.calibrations) < 2:
            print("Need at least 2 isotope calibrations for comparison")
            return
        
        fig, axes = plt.subplots(2, len(self.calibrations), figsize=(5*len(self.calibrations), 10))
        
        for idx, (isotope, data) in enumerate(self.calibrations.items()):
            # Plot correction map
            ax1 = axes[0, idx] if len(self.calibrations) > 1 else axes[0]
            im1 = ax1.imshow(data['correction_map'], cmap='RdYlGn', vmin=0.5, vmax=1.5)
            ax1.set_title(f'{isotope}\nCorrection Map', fontweight='bold')
            plt.colorbar(im1, ax=ax1)
            
            # Plot metrics
            ax2 = axes[1, idx] if len(self.calibrations) > 1 else axes[1]
            metrics = data['metrics']
            ax2.bar(['Integral\nUniformity', 'Differential\nUniformity'],
                   [metrics['integral_uniformity'], metrics['differential_uniformity']],
                   color=self.ISOTOPES[isotope]['color'], alpha=0.7)
            ax2.axhline(y=5, color='red', linestyle='--', label='NEMA Integral Limit')
            ax2.axhline(y=3, color='orange', linestyle='--', label='NEMA Diff Limit')
            ax2.set_ylabel('Uniformity (%)')
            ax2.set_title(f'{isotope}\nPerformance Metrics', fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.set_ylim(0, 10)
        
        plt.tight_layout()
        plt.savefig('multi_isotope_comparison.png', dpi=300, bbox_inches='tight')
        print("\nMulti-isotope comparison saved as 'multi_isotope_comparison.png'")
        plt.show()

class AutomatedQAReport:
    """Generate automated QA reports"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_report(self, 
                       flood_field: np.ndarray,
                       metrics: dict,
                       config: DetectorConfig,
                       filename: str = 'qa_report.html'):
        """Generate HTML QA report"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Determine pass/fail
        nema_pass = (metrics['integral_uniformity'] < 5.0 and 
                    metrics['differential_uniformity'] < 3.0)
        status_color = 'green' if nema_pass else 'red'
        status_text = 'PASS' if nema_pass else 'FAIL'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SPECT Detector QA Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ background-color: white; padding: 20px; margin-top: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #ecf0f1; border-radius: 3px; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
                .status-badge {{ display: inline-block; padding: 10px 20px; border-radius: 5px; 
                               background-color: {status_color}; color: white; font-size: 24px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SPECT Detector Quality Assurance Report</h1>
                <p>Generated: {timestamp}</p>
            </div>
            
            <div class="content">
                <h2>System Information</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Detector Size</td><td>{config.crystal_size[0]} x {config.crystal_size[1]} pixels</td></tr>
                    <tr><td>Pixel Size</td><td>{config.pixel_size} mm</td></tr>
                    <tr><td>Crystal Thickness</td><td>{config.crystal_thickness} mm NaI(Tl)</td></tr>
                    <tr><td>Energy Window</td><td>{config.energy_window[0]}-{config.energy_window[1]} keV</td></tr>
                    <tr><td>Total Counts</td><td>{np.sum(flood_field):,.0f}</td></tr>
                </table>
                
                <h2>Performance Metrics (NEMA NU 1-2018)</h2>
                <div class="metric">
                    <strong>Integral Uniformity (UFOV):</strong> {metrics['integral_uniformity']:.2f}%
                    <span class="{'pass' if metrics['integral_uniformity'] < 5.0 else 'fail'}">
                        ({'PASS' if metrics['integral_uniformity'] < 5.0 else 'FAIL'} - Limit: < 5.0%)
                    </span>
                </div>
                
                <div class="metric">
                    <strong>Differential Uniformity (UFOV):</strong> {metrics['differential_uniformity']:.2f}%
                    <span class="{'pass' if metrics['differential_uniformity'] < 3.0 else 'fail'}">
                        ({'PASS' if metrics['differential_uniformity'] < 3.0 else 'FAIL'} - Limit: < 3.0%)
                    </span>
                </div>
                
                <div class="metric">
                    <strong>Coefficient of Variation:</strong> {metrics.get('cv', 0):.2f}%
                </div>
                
                <h2>Overall Status</h2>
                <div style="text-align: center; margin: 30px 0;">
                    <span class="status-badge">{status_text}</span>
                </div>
                
                <h2>Recommendations</h2>
                <ul>
                    {'<li>System is operating within NEMA specifications. Continue daily QA.</li>' if nema_pass else 
                     '<li style="color: red;">System is OUT OF SPECIFICATION. Recalibration required.</li>' +
                     '<li>Perform flood field uniformity correction update.</li>' +
                     '<li>Check PMT gains and re-tune if necessary.</li>' +
                     '<li>Inspect collimator for damage or contamination.</li>'}
                </ul>
                
                <h2>Next QA Due</h2>
                <p><strong>Daily QA:</strong> {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}</p>
                <p><strong>Weekly QA:</strong> {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}</p>
                <p><strong>Quarterly PM:</strong> {(datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="content">
                <p style="text-align: center; color: #7f8c8d; font-size: 12px;">
                    This is an automated report generated by the SPECT Detector QA System<br>
                    For questions or concerns, contact Medical Physics or Service Engineering
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"\nAutomated QA report saved as '{filename}'")

def simulate_30_day_qa():
    """Simulate 30 days of QA measurements with realistic variation"""
    print("="*70)
    print("SIMULATING 30-DAY QA HISTORY")
    print("="*70)
    
    config = DetectorConfig()
    generator = FloodFieldGenerator(config)
    qa_tracker = TemporalQA()
    
    # Simulate gradual detector degradation
    start_date = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        
        # Gradual increase in non-uniformity (simulating PMT drift)
        degradation_factor = 1 + (day / 30) * 0.15  # 15% degradation over 30 days
        
        flood = generator.generate_flood_field(mean_counts=10000)
        
        # Add temporal drift
        flood *= degradation_factor
        
        # Calculate metrics
        metrics = {
            'integral_uniformity': PerformanceMetrics.calculate_integral_uniformity(flood),
            'differential_uniformity': PerformanceMetrics.calculate_differential_uniformity(flood),
            'nema_pass': PerformanceMetrics.calculate_integral_uniformity(flood) < 5.0
        }
        
        # Assign random technologist
        techs = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones']
        tech = np.random.choice(techs)
        
        qa_tracker.add_qa_measurement(current_date, metrics, tech)
        
        if day % 5 == 0:
            print(f"Day {day}: IU={metrics['integral_uniformity']:.2f}%, " + 
                  f"DU={metrics['differential_uniformity']:.2f}%, " +
                  f"Tech={tech}")
    
    # Generate trend report
    qa_tracker.generate_trend_report(days_back=30)

def main():
    """Demonstration of advanced QA capabilities"""
    
    print("="*70)
    print("ADVANCED SPECT DETECTOR QA ANALYSIS TOOLS")
    print("="*70)
    
    # 1. Temporal QA trending
    print("\n" + "="*70)
    print("1. TEMPORAL QA TRENDING")
    print("="*70)
    simulate_30_day_qa()
    
    # 2. Multi-isotope calibration
    print("\n" + "="*70)
    print("2. MULTI-ISOTOPE CALIBRATION")
    print("="*70)
    config = DetectorConfig()
    multi_iso = MultiIsotopeCalibration(config)
    
    for isotope in ['Tc-99m', 'I-123', 'Tl-201']:
        multi_iso.calibrate_isotope(isotope)
    
    multi_iso.compare_isotopes()
    
    # 3. Automated reporting
    print("\n" + "="*70)
    print("3. AUTOMATED QA REPORTING")
    print("="*70)
    
    generator = FloodFieldGenerator(config)
    flood = generator.generate_flood_field(mean_counts=10000)
    
    metrics = {
        'integral_uniformity': PerformanceMetrics.calculate_integral_uniformity(flood),
        'differential_uniformity': PerformanceMetrics.calculate_differential_uniformity(flood),
        'cv': (np.std(flood) / np.mean(flood)) * 100
    }
    
    reporter = AutomatedQAReport()
    reporter.generate_report(flood, metrics, config)
    
    print("\n" + "="*70)
    print("ADVANCED QA ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - qa_trend_report.png")
    print("  - multi_isotope_comparison.png")
    print("  - qa_report.html")

if __name__ == "__main__":
    main()