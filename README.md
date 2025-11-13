# SPECT Detector Calibration & Correction Pipeline

An interactive **React-based simulation** demonstrating the calibration and quality assurance pipeline of a **SPECT gamma camera detector**.
This web app models detector non-uniformities (PMT gain variations, edge effects, crystal defects) and applies **uniformity, energy, and linearity corrections**, providing real-time performance metrics and visualization.

---

##  Features

* **Synthetic Detector Simulation:**
  Generates a 64×64 grid mimicking a scintillation crystal array with random imperfections and photon statistics.

* **Calibration Modes:**

  * Raw (uncorrected data)
  * Uniformity correction
  * Energy correction
  * Linearity correction
  * Full calibration pipeline (all corrections)

* **Performance Metrics (NEMA NU-1 inspired):**

  * Integral uniformity (%)
  * Differential uniformity (%)
  * Energy resolution (% FWHM)
  * Spatial linearity distortion
  * Coefficient of variation (%)
  * Mean count rate

* **Visualization Dashboards:**

  * Flood field image (2D scatter)
  * Central line profile
  * Energy spectrum (Tc-99m)
  * Performance chart
  * Quality status indicator (color-coded)

* **Dynamic Interface:**

  * Smooth transitions
  * Gradient background (indigo → purple → pink)
  * TailwindCSS + Recharts visualizations
  * Lucide-react icons for clean, modern UI

---

##  Tech Stack

| Component     | Technology                               |
| ------------- | ---------------------------------------- |
| Frontend      | React.js (useState, useEffect)           |
| Visualization | [Recharts](https://recharts.org/en-US/)  |
| Styling       | [Tailwind CSS](https://tailwindcss.com/) |
| Icons         | [Lucide React](https://lucide.dev/)      |

---

##  Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/spect-detector-calibration.git
   cd spect-detector-calibration
   ```

2. **Install Dependencies**

   ```bash
   npm install
   ```

3. **Run the Development Server**

   ```bash
   npm start
   ```

   The app should open at [http://localhost:3000](http://localhost:3000).

4. **Build for Production**

   ```bash
   npm run build
   ```

---

##  How It Works

1. **Data Generation (`generateRawDetectorData`)**
   Simulates a detector flood field with:

   * PMT gain variations (4 quadrants)
   * Edge effects (up to 30% loss)
   * Random crystal defects (2%)
   * Poisson noise on photon counts
   * Random energy spread (~20 keV range)

2. **Correction Algorithms:**

   * **Uniformity Correction:** Normalizes pixel counts to mean detector response.
   * **Energy Correction:** Adjusts pixel energies based on radial drift from center.
   * **Linearity Correction:** Applies geometric barrel distortion compensation.

3. **Metric Calculation:**
   Computes NEMA NU-1 style metrics for uniformity, resolution, and stability.

4. **Visualization:**
   Uses Recharts to plot spatial maps, profiles, and histograms interactively.

---

##  Example Workflow

1. Start with **Raw Data (Uncorrected)** view.
2. Select **Uniformity**, **Energy**, or **Full Calibration** mode.
3. Click **Apply Calibration** ⚡
4. Observe changes in:

   * Flood field uniformity
   * Energy spectrum narrowing
   * Metric improvements
   * Updated “Quality Status” indicator

---

##  Key Functions

| Function                      | Description                           |
| ----------------------------- | ------------------------------------- |
| `generateRawDetectorData()`   | Simulates detector raw data grid      |
| `applyUniformityCorrection()` | Normalizes counts using mean response |
| `applyEnergyCorrection()`     | Compensates radial energy drift       |
| `applyLinearityCorrection()`  | Corrects geometric distortion         |
| `calculateMetrics()`          | Computes all performance parameters   |
| `runCalibration()`            | Executes chosen calibration mode      |
| `resetCalibration()`          | Resets detector to uncorrected state  |

---

## 🎨 Visualization Layout

* **Flood Field Image:**
  2D scatter plot of pixel intensities (`correctedCounts`)
* **Central Line Profile:**
  Count distribution along center row
* **Energy Spectrum:**
  Histogram of photon energies (Tc-99m)
* **Performance Chart:**
  Normalized calibration performance
* **Quality Indicator:**
  Displays system health:
  🟢 Excellent | 🔵 Good | 🟡 Acceptable | 🔴 Needs Calibration

---

## Scientific Background

This project simulates the **daily SPECT system calibration** performed in **clinical nuclear medicine** departments.
It mirrors correction and QA techniques applied in Siemens or GE systems to ensure accurate imaging and quantification.

Metrics follow principles from **NEMA NU-1 standards**, used to evaluate:

* Flood field uniformity
* Energy and spatial resolution
* Linearity across detector surfaces

---

##  Future Extensions

* Import real flood field images (.csv / DICOM)
* Integrate Monte Carlo simulations for photon interaction modeling
* Add noise models (Gaussian, electronic)
* Enable calibration comparison (pre/post)
* Export calibration reports

---

##  Author

**Abraham Taye**
Biomedical Engineering Graduate Student, Marquette University & Medical College of Wisconsin
Research: Medical Image Processing • Computational Modeling • SPECT Imaging
📧 [LinkedIn](https://www.linkedin.com/in/abrahamtaye) | 🌐 [Portfolio](https://github.com/abrahamgenetu)

---

Would you like me to make the README more **GitHub-styled** (with badges, emojis, and a quick “Run Demo” section)?
I can make it visually polished and ready for your repo page.
