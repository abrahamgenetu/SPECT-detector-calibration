import React, { useState, useEffect } from 'react';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Zap, Settings, CheckCircle, AlertTriangle } from 'lucide-react';

const DetectorCalibration = () => {
  const [calibrationStep, setCalibrationStep] = useState('raw');
  const [crystalMap, setCrystalMap] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Generate synthetic detector data with non-uniformities
  const generateRawDetectorData = () => {
    const size = 64;
    const data = [];
    
    // Simulate PMT gain variations and edge effects
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const distFromCenter = Math.sqrt(Math.pow(x - size/2, 2) + Math.pow(y - size/2, 2));
        const edgeEffect = 1.0 - (distFromCenter / (size/2)) * 0.3; // 30% drop at edges
        
        // PMT non-uniformity (4 PMT quadrants with different gains)
        const pmtGain = (x < size/2 && y < size/2) ? 1.0 :
                       (x >= size/2 && y < size/2) ? 0.85 :
                       (x < size/2 && y >= size/2) ? 0.92 :
                       0.88;
        
        // Crystal defects (random dead spots)
        const isDefective = Math.random() < 0.02;
        const defectFactor = isDefective ? 0.3 : 1.0;
        
        // Base count rate with Poisson noise
        const baseCount = 1000;
        const poissonNoise = Math.sqrt(baseCount) * (Math.random() - 0.5) * 2;
        
        const counts = Math.max(0, baseCount * edgeEffect * pmtGain * defectFactor + poissonNoise);
        
        data.push({
          x, y,
          rawCounts: counts,
          correctedCounts: counts,
          energy: 140 + (Math.random() - 0.5) * 20 // keV with spread
        });
      }
    }
    
    return data;
  };

  // Flood field uniformity correction
  const applyUniformityCorrection = (data) => {
    // Calculate mean count
    const meanCount = data.reduce((sum, d) => sum + d.rawCounts, 0) / data.length;
    
    // Generate correction factors
    return data.map(pixel => {
      const correctionFactor = meanCount / (pixel.rawCounts || 1);
      return {
        ...pixel,
        correctedCounts: pixel.rawCounts * Math.min(correctionFactor, 3.0), // Cap at 3x
        correctionFactor
      };
    });
  };

  // Energy correction based on position
  const applyEnergyCorrection = (data) => {
    const size = 64;
    return data.map(pixel => {
      // Energy drift correction (position-dependent)
      const distFromCenter = Math.sqrt(
        Math.pow(pixel.x - size/2, 2) + 
        Math.pow(pixel.y - size/2, 2)
      );
      const energyDrift = (distFromCenter / (size/2)) * 5; // 5 keV drift
      
      return {
        ...pixel,
        correctedEnergy: pixel.energy + energyDrift
      };
    });
  };

  // Linearity correction (spatial distortion)
  const applyLinearityCorrection = (data) => {
    return data.map(pixel => {
      // Apply barrel distortion correction
      const centerX = 32, centerY = 32;
      const dx = pixel.x - centerX;
      const dy = pixel.y - centerY;
      const r = Math.sqrt(dx*dx + dy*dy);
      
      // Distortion model
      const k = 0.002; // distortion coefficient
      const distortion = 1 + k * r * r;
      
      return {
        ...pixel,
        correctedX: centerX + dx / distortion,
        correctedY: centerY + dy / distortion
      };
    });
  };

  // Calculate performance metrics
  const calculateMetrics = (data) => {
    const counts = data.map(d => d.correctedCounts);
    const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
    const variance = counts.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / counts.length;
    const stdDev = Math.sqrt(variance);
    
    // Integral uniformity
    const maxCount = Math.max(...counts);
    const minCount = Math.min(...counts);
    const integralUniformity = ((maxCount - minCount) / (maxCount + minCount)) * 100;
    
    // Differential uniformity (max difference in adjacent pixels)
    let maxDiff = 0;
    for (let i = 0; i < data.length - 1; i++) {
      if (data[i].y === data[i+1].y) { // same row
        const diff = Math.abs(data[i].correctedCounts - data[i+1].correctedCounts);
        maxDiff = Math.max(maxDiff, diff);
      }
    }
    const differentialUniformity = (maxDiff / mean) * 100;
    
    // Energy resolution
    const energies = data.map(d => d.energy);
    const energyMean = energies.reduce((a, b) => a + b, 0) / energies.length;
    const energyStdDev = Math.sqrt(
      energies.reduce((sum, e) => sum + Math.pow(e - energyMean, 2), 0) / energies.length
    );
    const energyResolution = (2.355 * energyStdDev / energyMean) * 100; // FWHM %
    
    // Spatial resolution (simplified - linearity measure)
    const spatialDistortions = data.map(d => {
      if (d.correctedX && d.correctedY) {
        return Math.sqrt(
          Math.pow(d.x - d.correctedX, 2) + 
          Math.pow(d.y - d.correctedY, 2)
        );
      }
      return 0;
    });
    const avgDistortion = spatialDistortions.reduce((a, b) => a + b, 0) / spatialDistortions.length;
    
    return {
      integralUniformity: integralUniformity.toFixed(2),
      differentialUniformity: differentialUniformity.toFixed(2),
      energyResolution: energyResolution.toFixed(2),
      spatialLinearity: avgDistortion.toFixed(3),
      coefficientOfVariation: ((stdDev / mean) * 100).toFixed(2),
      meanCount: mean.toFixed(0)
    };
  };

  // Initialize with raw data
  useEffect(() => {
    const rawData = generateRawDetectorData();
    setCrystalMap(rawData);
    setPerformanceMetrics(calculateMetrics(rawData));
  }, []);

  const runCalibration = () => {
    setIsProcessing(true);
    
    setTimeout(() => {
      let data = [...crystalMap];
      
      if (calibrationStep === 'uniformity' || calibrationStep === 'all') {
        data = applyUniformityCorrection(data);
      }
      if (calibrationStep === 'energy' || calibrationStep === 'all') {
        data = applyEnergyCorrection(data);
      }
      if (calibrationStep === 'linearity' || calibrationStep === 'all') {
        data = applyLinearityCorrection(data);
      }
      
      setCrystalMap(data);
      setPerformanceMetrics(calculateMetrics(data));
      setIsProcessing(false);
    }, 800);
  };

  const resetCalibration = () => {
    const rawData = generateRawDetectorData();
    setCrystalMap(rawData);
    setPerformanceMetrics(calculateMetrics(rawData));
    setCalibrationStep('raw');
  };

  // Prepare heatmap data
  const getHeatmapData = () => {
    const sampledData = crystalMap.filter((_, idx) => idx % 16 === 0); // Sample for performance
    return sampledData.map(d => ({
      x: d.x,
      y: d.y,
      value: d.correctedCounts
    }));
  };

  // Line profile data
  const getLineProfile = () => {
    const centerRow = crystalMap.filter(d => d.y === 32);
    return centerRow.map(d => ({
      position: d.x,
      counts: d.correctedCounts
    }));
  };

  // Energy spectrum data
  const getEnergySpectrum = () => {
    const histogram = {};
    crystalMap.forEach(d => {
      const bin = Math.round(d.energy);
      histogram[bin] = (histogram[bin] || 0) + 1;
    });
    
    return Object.keys(histogram)
      .sort((a, b) => a - b)
      .map(energy => ({
        energy: parseInt(energy),
        counts: histogram[energy]
      }));
  };

  // Performance comparison data
  const getPerformanceData = () => {
    if (!performanceMetrics) return [];
    
    return [
      {
        metric: 'Uniformity',
        value: Math.max(0, 100 - parseFloat(performanceMetrics.integralUniformity))
      },
      {
        metric: 'Energy',
        value: Math.max(0, 100 - parseFloat(performanceMetrics.energyResolution) * 5)
      },
      {
        metric: 'Linearity',
        value: Math.max(0, 100 - parseFloat(performanceMetrics.spatialLinearity) * 20)
      },
      {
        metric: 'Stability',
        value: Math.max(0, 100 - parseFloat(performanceMetrics.coefficientOfVariation))
      }
    ];
  };

  const getQualityStatus = () => {
   if (!performanceMetrics) return { status: 'unknown', color: 'gray', icon: Settings };

    
    const uniformity = parseFloat(performanceMetrics.integralUniformity);
    const energyRes = parseFloat(performanceMetrics.energyResolution);
    
    if (uniformity < 5 && energyRes < 10) {
      return { status: 'Excellent', color: 'green', icon: CheckCircle };
    } else if (uniformity < 10 && energyRes < 12) {
      return { status: 'Good', color: 'blue', icon: CheckCircle };
    } else if (uniformity < 15 && energyRes < 15) {
      return { status: 'Acceptable', color: 'yellow', icon: AlertTriangle };
    } else {
      return { status: 'Needs Calibration', color: 'red', icon: AlertTriangle };
    }
  };

  const quality = getQualityStatus();
  const QualityIcon = quality.icon;

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Settings className="w-10 h-10" />
            SPECT Detector Calibration & Correction Pipeline
          </h1>
          <p className="text-purple-200 text-lg">Quality Assurance & Image Optimization System</p>
        </div>

        {/* Control Panel */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6 border border-white/20">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="text-purple-200 text-sm block mb-3 font-semibold">Calibration Mode</label>
              <div className="space-y-2">
                {[
                  { value: 'raw', label: 'Raw Data (Uncorrected)' },
                  { value: 'uniformity', label: 'Uniformity Correction' },
                  { value: 'energy', label: 'Energy Correction' },
                  { value: 'linearity', label: 'Linearity Correction' },
                  { value: 'all', label: 'Full Calibration Pipeline' }
                ].map(option => (
                  <label key={option.value} className="flex items-center text-white cursor-pointer">
                    <input
                      type="radio"
                      value={option.value}
                      checked={calibrationStep === option.value}
                      onChange={(e) => setCalibrationStep(e.target.value)}
                      className="mr-3"
                    />
                    {option.label}
                  </label>
                ))}
              </div>
            </div>

            <div className="flex flex-col justify-between">
              <div className="mb-4">
                <label className="text-purple-200 text-sm block mb-2 font-semibold">Quality Status</label>
                <div className={`flex items-center gap-2 text-${quality.color}-400 text-xl font-bold`}>
                  <QualityIcon className="w-8 h-8" />
                  {quality.status}
                </div>
              </div>

              <div className="space-y-2">
                <button
                  onClick={runCalibration}
                  disabled={isProcessing || calibrationStep === 'raw'}
                  className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-purple-700 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  <Zap className="w-5 h-5" />
                  {isProcessing ? 'Processing...' : 'Apply Calibration'}
                </button>
                
                <button
                  onClick={resetCalibration}
                  className="w-full bg-white/10 text-white py-2 rounded-lg font-semibold hover:bg-white/20 transition-all border border-white/30"
                >
                  Reset to Raw Data
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Dashboard */}
        {performanceMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            {Object.entries(performanceMetrics).map(([key, value]) => (
              <div key={key} className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-white/20">
                <div className="text-purple-200 text-xs mb-1 uppercase">
                  {key.replace(/([A-Z])/g, ' $1').trim()}
                </div>
                <div className="text-2xl font-bold text-white">
                  {value}
                  {key.includes('Uniformity') || key.includes('Resolution') || key.includes('Variation') ? '%' : ''}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Visualizations */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Detector Flood Image */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h3 className="text-lg font-bold text-white mb-4">Flood Field Image</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  domain={[0, 64]}
                  stroke="#a78bfa"
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  domain={[0, 64]}
                  stroke="#a78bfa"
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#1e1b4b', border: '1px solid #8b5cf6' }}
                />
                <Scatter 
                  data={getHeatmapData()} 
                  fill="#8b5cf6" 
                  fillOpacity={0.8}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Line Profile */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h3 className="text-lg font-bold text-white mb-4">Central Line Profile</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={getLineProfile()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis 
                  dataKey="position" 
                  label={{ value: 'Pixel Position', position: 'insideBottom', offset: -10, fill: '#a78bfa' }}
                  stroke="#a78bfa"
                />
                <YAxis 
                  label={{ value: 'Counts', angle: -90, position: 'insideLeft', fill: '#a78bfa' }}
                  stroke="#a78bfa"
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e1b4b', border: '1px solid #8b5cf6' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="counts" 
                  stroke="#8b5cf6" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Energy Spectrum */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h3 className="text-lg font-bold text-white mb-4">Energy Spectrum (Tc-99m)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={getEnergySpectrum()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis 
                  dataKey="energy" 
                  label={{ value: 'Energy (keV)', position: 'insideBottom', offset: -10, fill: '#a78bfa' }}
                  stroke="#a78bfa"
                  domain={[120, 160]}
                />
                <YAxis 
                  label={{ value: 'Counts', angle: -90, position: 'insideLeft', fill: '#a78bfa' }}
                  stroke="#a78bfa"
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e1b4b', border: '1px solid #8b5cf6' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="counts" 
                  stroke="#ec4899" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Performance Bar Chart */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h3 className="text-lg font-bold text-white mb-4">Overall Performance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={getPerformanceData()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis 
                  dataKey="metric" 
                  stroke="#a78bfa"
                />
                <YAxis 
                  domain={[0, 100]}
                  label={{ value: 'Performance Score', angle: -90, position: 'insideLeft', fill: '#a78bfa' }}
                  stroke="#a78bfa"
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e1b4b', border: '1px solid #8b5cf6' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#8b5cf6" 
                  strokeWidth={3}
                  fill="#8b5cf6"
                  fillOpacity={0.6}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Technical Info */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
          <h3 className="text-lg font-bold text-white mb-3">About This Calibration System</h3>
          <p className="text-purple-200 leading-relaxed">
            This simulation demonstrates a complete SPECT detector quality assurance and calibration pipeline. 
            It models realistic detector non-uniformities including PMT gain variations, edge effects, and crystal 
            defects, then applies industry-standard correction algorithms. The system calculates NEMA NU-1 performance 
            metrics including integral and differential uniformity, energy resolution, and spatial linearity. This 
            type of calibration is performed daily in clinical nuclear medicine departments and is critical for 
            maintaining image quality standards required by Siemens SPECT systems.
          </p>
        </div>
      </div>
    </div>
  );
};

export default DetectorCalibration;