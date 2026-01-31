import { useRef } from 'react';
import Plot from 'react-plotly.js';
import { Image, FileText } from 'lucide-react';
import { api } from '../services/api';

interface ChartDisplayProps {
  chart: {
    id: string;
    type: string;
    title: string;
    plotly_json?: any;
    plotlyJson?: any;
  };
}

export function ChartDisplay({ chart }: ChartDisplayProps) {
  const plotRef = useRef<any>(null);
  
  const plotlyData = chart.plotly_json || chart.plotlyJson;
  
  if (!plotlyData) {
    return (
      <div className="p-4 text-center text-gray-500">
        Unable to render chart
      </div>
    );
  }

  const handleDownloadPNG = async () => {
    try {
      const blob = await api.exportChartPng(chart.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${chart.title.replace(/\s+/g, '_')}.png`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      // Fallback: use Plotly's built-in download
      if (plotRef.current) {
        const Plotly = (window as any).Plotly;
        if (Plotly) {
          Plotly.downloadImage(plotRef.current.el, {
            format: 'png',
            filename: chart.title.replace(/\s+/g, '_'),
            width: 1200,
            height: 800,
          });
        }
      }
    }
  };

  const handleDownloadData = async () => {
    try {
      const blob = await api.exportChartData(chart.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${chart.title.replace(/\s+/g, '_')}_data.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  return (
    <div className="plotly-chart">
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-medium text-gray-900">{chart.title}</h3>
        <div className="flex gap-1">
          <button
            onClick={handleDownloadPNG}
            className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
            title="Download as PNG"
          >
            <Image className="w-4 h-4" />
          </button>
          <button
            onClick={handleDownloadData}
            className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
            title="Download data as CSV"
          >
            <FileText className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Plotly Chart */}
      <Plot
        ref={plotRef}
        data={plotlyData.data}
        layout={{
          ...plotlyData.layout,
          autosize: true,
          margin: { l: 50, r: 30, t: 30, b: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { family: 'Inter, sans-serif' },
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        }}
        style={{ width: '100%', height: '350px' }}
        useResizeHandler
      />
    </div>
  );
}
