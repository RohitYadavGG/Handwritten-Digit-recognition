// ============================================
// PERFORMANCE CHARTS INITIALIZATION
// ============================================

// Function to get chart colors based on theme
function getChartColors() {
  const theme = document.body.getAttribute('data-theme') || 'dark';
  
  if (theme === 'light') {
    return {
      textColor: '#333333',
      titleColor: '#1a1a1a',
      bgColor: '#ffffff',
      gridColor: 'rgba(0, 0, 0, 0.1)',
      markerStroke: '#f0f0f0',
      tooltipTheme: 'light'
    };
  }
  
  return {
    textColor: '#b0b0b0',
    titleColor: '#ffffff',
    bgColor: 'transparent',
    gridColor: 'rgba(255, 255, 255, 0.08)',
    markerStroke: '#1a1a1a',
    tooltipTheme: 'dark'
  };
}

// Sample real data from model training (these can be replaced with actual model data)
const modelPerformanceData = {
  // Per-class accuracy for each digit (0-9) - ACTUAL TEST DATA
  perClassAccuracy: [
    { digit: '0', accuracy: 99.8 },
    { digit: '1', accuracy: 99.65 },
    { digit: '2', accuracy: 99.32 },
    { digit: '3', accuracy: 99.31 },
    { digit: '4', accuracy: 99.8 },
    { digit: '5', accuracy: 99.44 },
    { digit: '6', accuracy: 99.06 },
    { digit: '7', accuracy: 99.32 },
    { digit: '8', accuracy: 98.97 },
    { digit: '9', accuracy: 98.91 }
  ],
  
  // Overall metrics: Precision, Recall, F1-Score, Accuracy - ACTUAL DATA
  overallMetrics: {
    names: ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
    values: [99.36, 99.36, 99.36, 99.36]
  },
  
  // Training history over epochs (simulated - based on typical CNN training)
  trainingHistory: {
    epochs: [1, 5, 10, 15, 20, 25, 30],
    trainLoss: [1.95, 0.52, 0.22, 0.11, 0.07, 0.04, 0.02],
    trainAccuracy: [18, 78, 90, 95, 97, 98, 99],
    valLoss: [2.1, 0.65, 0.28, 0.15, 0.09, 0.06, 0.04],
    valAccuracy: [15, 75, 88, 93, 96, 97.5, 99]
  },
  
  // Per-digit F1 scores (ACTUAL TEST DATA)
  perDigitF1: [99.59, 99.6, 99.23, 99.45, 99.39, 99.33, 99.27, 99.17, 99.28, 99.25],
  
  // Average confidence per digit (ACTUAL TEST DATA)
  avgConfidencePerDigit: [99.74, 99.57, 99.24, 99.48, 99.79, 99.5, 99.19, 99.69, 99.19, 98.38],
  
  // Confusion matrix (ACTUAL TEST DATA - 10000 MNIST test images)
  confusionMatrix: [
    [978, 0, 2, 0, 0, 1, 1, 0, 2, 0],
    [0, 1131, 0, 0, 0, 0, 3, 1, 0, 1],
    [0, 1, 1025, 1, 0, 0, 0, 4, 3, 0],
    [0, 0, 0, 1003, 0, 3, 0, 0, 0, 1],
    [0, 0, 1, 0, 980, 0, 2, 1, 1, 5],
    [0, 0, 0, 4, 0, 887, 2, 0, 1, 0],
    [2, 1, 0, 0, 0, 1, 949, 0, 1, 0],
    [0, 2, 4, 1, 0, 0, 0, 1021, 0, 3],
    [0, 0, 0, 1, 0, 0, 1, 1, 964, 1],
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 998]
  ]
};

// ============================================
// 1. PER-CLASS ACCURACY CHART (TREND LINE)
// ============================================
function initPerClassChart() {
  const digits = modelPerformanceData.perClassAccuracy.map(d => d.digit);
  const accuracies = modelPerformanceData.perClassAccuracy.map(d => d.accuracy);
  
  const perClassOptions = {
    chart: {
      height: 300,
      type: 'line',
      fontFamily: 'Helvetica, Arial, sans-serif',
      foreColor: getChartColors().textColor,
      background: getChartColors().bgColor,
      toolbar: {
        show: false,
      },
      animations: {
        enabled: true,
        speed: 800,
        animateGradually: {
          enabled: true,
          delay: 150
        }
      }
    },
    stroke: {
      curve: 'smooth',
      width: 3,
      lineCap: 'round'
    },
    series: [
      {
        name: 'Accuracy (%)',
        data: accuracies,
      }
    ],
    title: {
      text: 'Per-Digit Recognition Accuracy Trend',
      align: 'left',
      offsetX: 0,
      offsetY: 0,
      style: {
        fontSize: '14px',
        fontWeight: 'bold',
        color: getChartColors().titleColor,
      },
    },
    dataLabels: {
      enabled: false
    },
    markers: {
      size: 5,
      strokeWidth: 2,
      hover: {
        size: 7,
      },
      colors: ['#ff006e'],
      strokeColor: getChartColors().markerStroke
    },
    fill: {
      type: 'gradient',
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.45,
        opacityTo: 0.05,
        stops: [20, 100, 100, 100],
        colorStops: [
          {
            offset: 0,
            color: '#00d4ff',
            opacity: 0.8
          },
          {
            offset: 100,
            color: '#667dff',
            opacity: 0.1
          }
        ]
      }
    },
    colors: ['#00d4ff'],
    xaxis: {
      categories: digits.map(d => `Digit ${d}`),
      title: {
        text: 'Digit Class',
        style: {
          color: getChartColors().textColor,
          fontSize: '12px'
        }
      }
    },
    yaxis: {
      title: {
        text: 'Accuracy (%)',
        style: {
          color: getChartColors().textColor,
          fontSize: '12px'
        }
      },
      min: 95,
      max: 100
    },
    grid: {
      borderColor: getChartColors().gridColor,
      xaxis: {
        lines: {
          show: true,
        }
      }
    },
    tooltip: {
      y: {
        formatter: function(val) {
          return val.toFixed(2) + '%';
        }
      },
      theme: getChartColors().tooltipTheme
    },
    responsive: [{
      breakpoint: 600,
      options: {
        chart: {
          height: 280
        }
      }
    }]
  };

  const perClassChart = new ApexCharts(document.querySelector('#per-class-chart'), perClassOptions);
  perClassChart.render();
}

// ============================================
// 2. OVERALL METRICS CHART
// ============================================
function initMetricsChart() {
  const colors = getChartColors();
  const metricsOptions = {
    chart: {
      height: 350,
      type: 'radar',
      fontFamily: 'Helvetica, Arial, sans-serif',
      foreColor: colors.textColor,
      toolbar: {
        show: false,
      }
    },
    series: [
      {
        name: 'Model Performance',
        data: modelPerformanceData.overallMetrics.values
      }
    ],
    title: {
      text: 'Overall Model Performance Metrics',
      align: 'left',
      offsetX: 0,
      offsetY: 0,
      style: {
        fontSize: '14px',
        fontWeight: 'bold',
        color: colors.titleColor,
      }
    },
    xaxis: {
      categories: modelPerformanceData.overallMetrics.names
    },
    plotOptions: {
      radar: {
        polygons: {
          strokeColors: colors.gridColor,
          fill: {
            colors: ['rgba(102, 125, 255, 0.1)', 'rgba(0, 212, 255, 0.05)']
          }
        }
      }
    },
    stroke: {
      show: true,
      width: 2,
      colors: ['#ff006e'],
      dashArray: 0
    },
    fill: {
      opacity: 0.5,
      colors: ['#ff006e']
    },
    markers: {
      size: 5,
      colors: ['#00d4ff'],
      strokeColor: colors.markerStroke,
      strokeWidth: 2
    },
    tooltip: {
      y: {
        formatter: function(val) {
          return val.toFixed(2) + '%';
        }
      },
      theme: colors.tooltipTheme
    },
    grid: {
      show: true
    },
    responsive: [{
      breakpoint: 600,
      options: {
        chart: {
          height: 300
        }
      }
    }]
  };

  const metricsChart = new ApexCharts(document.querySelector('#metrics-chart'), metricsOptions);
  metricsChart.render();
}

// ============================================
// 3. TRAINING HISTORY CHART
// ============================================
function initTrainingChart() {
  const colors = getChartColors();
  const trainingOptions = {
    chart: {
      height: 350,
      type: 'line',
      fontFamily: 'Helvetica, Arial, sans-serif',
      foreColor: colors.textColor,
      toolbar: {
        show: false,
      },
      animations: {
        enabled: true,
        speed: 800,
      }
    },
    stroke: {
      curve: 'smooth',
      width: 3,
    },
    series: [
      {
        name: 'Training Accuracy',
        data: modelPerformanceData.trainingHistory.trainAccuracy,
      },
      {
        name: 'Validation Accuracy',
        data: modelPerformanceData.trainingHistory.valAccuracy,
      }
    ],
    title: {
      text: 'Training Progress Over Epochs',
      align: 'left',
      offsetX: 0,
      offsetY: 0,
      style: {
        fontSize: '14px',
        fontWeight: 'bold',
        color: colors.titleColor,
      }
    },
    markers: {
      size: 5,
      strokeWidth: 2,
      hover: {
        size: 7,
      },
      colors: ['#00d4ff', '#ffd60a'],
      strokeColor: colors.markerStroke
    },
    grid: {
      show: true,
      borderColor: colors.gridColor,
      padding: {
        bottom: 0,
      },
    },
    labels: modelPerformanceData.trainingHistory.epochs,
    xaxis: {
      title: {
        text: 'Epoch',
        style: {
          color: colors.textColor,
          fontSize: '12px'
        }
      },
      tooltip: {
        enabled: false,
      },
    },
    yaxis: {
      title: {
        text: 'Accuracy (%)',
        style: {
          color: colors.textColor,
          fontSize: '12px'
        }
      }
    },
    legend: {
      position: 'top',
      horizontalAlign: 'right',
      offsetY: -10,
      labels: {
        colors: colors.textColor,
      },
    },
    colors: ['#00d4ff', '#ff006e'],
    tooltip: {
      y: {
        formatter: function(val) {
          return val.toFixed(2) + '%';
        }
      },
      theme: colors.tooltipTheme
    },
    responsive: [{
      breakpoint: 600,
      options: {
        chart: {
          height: 300
        }
      }
    }]
  };

  const trainingChart = new ApexCharts(document.querySelector('#training-chart'), trainingOptions);
  trainingChart.render();
}

// ============================================
// 4. CONFUSION MATRIX HEATMAP
// ============================================
function initConfusionChart() {
  // Prepare confusion matrix data for heatmap
  const confusionData = [];
  const digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) {
      confusionData.push({
        x: `Actual: ${digits[j]}`,
        y: `Predicted: ${digits[i]}`,
        value: modelPerformanceData.confusionMatrix[i][j]
      });
    }
  }
  
  // Create normalized data for color intensity (0-100)
  const maxValue = Math.max(...confusionData.map(d => d.value));
  const normalizedData = confusionData.map(d => ({
    ...d,
    normalized: (d.value / maxValue) * 100
  }));
  
  // Prepare series data
  const seriesData = [];
  for (let i = 0; i < 10; i++) {
    const row = [];
    for (let j = 0; j < 10; j++) {
      row.push(modelPerformanceData.confusionMatrix[i][j]);
    }
    seriesData.push({
      name: `Predicted: ${digits[i]}`,
      data: row
    });
  }
  
  const confusionOptions = {
    chart: {
      height: 350,
      type: 'heatmap',
      fontFamily: 'Helvetica, Arial, sans-serif',
      foreColor: '#b0b0b0',
      toolbar: {
        show: false,
      }
    },
    plotOptions: {
      heatmap: {
        shadeIntensity: 0.5,
        radius: 0,
        useFillColorAsStroke: true,
      }
    },
    dataLabels: {
      enabled: false
    },
    series: [
      {
        name: 'Confusion Matrix',
        data: normalizedData.slice(0, 10).map((d, i) => ({
          x: 0,
          y: d.normalized
        }))
      }
    ],
    title: {
      text: 'Confusion Matrix Heatmap',
      align: 'left',
      offsetX: 0,
      offsetY: 0,
      style: {
        fontSize: '14px',
        fontWeight: 'bold',
        color: '#ffffff',
      }
    },
    xaxis: {
      categories: digits.map(d => `A${d}`),
      title: {
        text: 'Actual Digit',
        style: {
          color: '#b0b0b0',
          fontSize: '12px'
        }
      }
    },
    yaxis: {
      title: {
        text: 'Predicted Digit',
        style: {
          color: '#b0b0b0',
          fontSize: '12px'
        }
      }
    },
    colors: ['#667dff', '#00d4ff', '#ff006e'],
    tooltip: {
      theme: 'dark'
    }
  };

  // Create a custom heatmap visual using a compact table-like structure
  const confusionContainer = document.querySelector('#confusion-chart');
  
  let htmlContent = `
    <div style="overflow-x: auto; display: flex; justify-content: center; padding: 6px;">
      <table style="border-collapse: collapse; font-size: 8px;">
        <thead>
          <tr>
            <th style="border: 1px solid rgba(255, 255, 255, 0.1); padding: 1px; background: var(--bg-tertiary); text-align: center; width: 20px;">P/A</th>
            ${digits.map(d => `<th style="border: 1px solid rgba(255, 255, 255, 0.1); padding: 1px; background: var(--bg-tertiary); color: #667dff; width: 20px; text-align: center;">${d}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
  `;
  
  for (let i = 0; i < 10; i++) {
    htmlContent += `
      <tr>
        <td style="border: 1px solid rgba(255, 255, 255, 0.1); padding: 1px; background: var(--bg-tertiary); color: #667dff; font-weight: bold; text-align: center; width: 20px;">${i}</td>
    `;
    
    for (let j = 0; j < 10; j++) {
      const value = modelPerformanceData.confusionMatrix[i][j];
      const maxVal = Math.max(...modelPerformanceData.confusionMatrix.flat());
      const intensity = value / maxVal;
      
      // Determine background color intensity (darker = higher accuracy)
      let bgColor;
      if (i === j) {
        // On-diagonal (correct predictions)
        bgColor = `rgba(102, 125, 255, ${0.3 + intensity * 0.5})`;
      } else {
        // Off-diagonal (misclassifications)
        bgColor = `rgba(255, 0, 110, ${intensity * 0.2})`;
      }
      
      htmlContent += `
        <td style="
          border: 1px solid rgba(255, 255, 255, 0.05); 
          padding: 1px; 
          background: ${bgColor};
          text-align: center;
          color: #ffffff;
          font-size: 7px;
          width: 20px;
          height: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
        " title="Pred: ${i}, Act: ${j}, Count: ${value}">
          ${value}
        </td>
      `;
    }
    
    htmlContent += `</tr>`;
  }
  
  htmlContent += `
        </tbody>
      </table>
    </div>
    <div style="text-align: center; font-size: 10px; color: #b0b0b0; margin-top: 6px; padding: 0 5px;">
      <p style="margin: 0;"><span style="color: #667dff;">■</span> Correct | <span style="color: #ff006e;">■</span> Misclassified</p>
    </div>
  `;
  
  confusionContainer.innerHTML = htmlContent;
}

// ============================================
// INITIALIZE ALL CHARTS ON PAGE LOAD
// ============================================

// Function to safely setup chart observers for lazy loading
function setupChartObservers() {
  // Check if ApexCharts is available
  if (typeof ApexCharts === 'undefined') {
    console.error('ApexCharts library not loaded');
    setTimeout(setupChartObservers, 500);
    return;
  }
  
  const chartConfigs = [
    { id: 'per-class-chart', initFn: initPerClassChart },
    { id: 'metrics-chart', initFn: initMetricsChart },
    { id: 'training-chart', initFn: initTrainingChart }
  ];
  
  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const chartId = entry.target.id;
        const config = chartConfigs.find(c => c.id === chartId);
        if (config) {
          try {
            config.initFn();
            console.log(`${chartId} initialized via scroll`);
            obs.unobserve(entry.target);
          } catch (e) {
            console.error(`Error initializing ${chartId}:`, e);
          }
        }
      }
    });
  }, { threshold: 0.2 }); // Trigger when 20% visible
  
  chartConfigs.forEach(config => {
    const el = document.getElementById(config.id);
    if (el) observer.observe(el);
  });
}

// Setup observers when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    setTimeout(setupChartObservers, 300);
  });
} else {
  setTimeout(setupChartObservers, 300);
}