import { useState } from 'react';
import { 
  BarChart2, 
  LineChart, 
  PieChart, 
  Activity,
  TrendingUp,
  Box
} from 'lucide-react';
import Plot from 'react-plotly.js';
import { useStore } from '../store/useStore';
import { api } from '../services/api';

type ChartType = 'bar' | 'line' | 'pie' | 'scatter' | 'histogram' | 'box';

const CHART_TYPES: { type: ChartType; icon: any; label: string }[] = [
  { type: 'bar', icon: BarChart2, label: 'Bar' },
  { type: 'line', icon: LineChart, label: 'Line' },
  { type: 'pie', icon: PieChart, label: 'Pie' },
  { type: 'scatter', icon: Activity, label: 'Scatter' },
  { type: 'histogram', icon: TrendingUp, label: 'Histogram' },
  { type: 'box', icon: Box, label: 'Box Plot' },
];

const AGGREGATIONS = [
  { value: 'sum', label: 'Sum' },
  { value: 'mean', label: 'Average' },
  { value: 'count', label: 'Count' },
  { value: 'min', label: 'Min' },
  { value: 'max', label: 'Max' },
];

export function QuickChartsPanel() {
  const { activeDataset } = useStore();
  const [chartType, setChartType] = useState<ChartType>('bar');
  const [xColumn, setXColumn] = useState('');
  const [yColumn, setYColumn] = useState('');
  const [colorColumn, setColorColumn] = useState('');
  const [aggregation, setAggregation] = useState('sum');
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(false);
  const [chart, setChart] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  if (!activeDataset) return null;

  const columns = activeDataset.columns || [];
  const numericColumns = activeDataset.numericColumns || (activeDataset as any).numeric_columns || [];
  const categoricalColumns = columns.filter((c: string) => !numericColumns.includes(c));
  const yAxisColumns = numericColumns.length > 0 ? numericColumns : columns;

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.createQuickChart({
        chartType,
        xColumn: xColumn || undefined,
        yColumn: yColumn || undefined,
        colorColumn: colorColumn || undefined,
        title: title || undefined,
        aggregation,
      });
      setChart(result.chart);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate chart');
    } finally {
      setLoading(false);
    }
  };

  const needsXY = ['bar', 'line', 'scatter', 'pie'].includes(chartType);
  const needsAggregation = ['bar', 'pie'].includes(chartType);

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Chart Type Selection */}
      <div className="p-3 border-b">
        <label className="block text-xs font-medium text-gray-500 mb-2">Chart Type</label>
        <div className="grid grid-cols-3 gap-1">
          {CHART_TYPES.map(({ type, icon: Icon, label }) => (
            <button
              key={type}
              onClick={() => setChartType(type)}
              className={`flex flex-col items-center p-2 rounded-lg text-xs transition-colors ${
                chartType === type
                  ? 'bg-primary-100 text-primary-700 border border-primary-300'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-transparent'
              }`}
            >
              <Icon className="w-4 h-4 mb-1" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Column Selection */}
      <div className="p-3 border-b space-y-3">
        {/* X Column */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">
            {chartType === 'pie' ? 'Names (Categories)' : chartType === 'histogram' ? 'Column' : 'X Axis'}
          </label>
          <select
            value={xColumn}
            onChange={(e) => setXColumn(e.target.value)}
            className="w-full px-2 py-1.5 text-sm border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select column...</option>
            {columns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>

        {/* Y Column */}
        {needsXY && chartType !== 'histogram' && (
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">
              {chartType === 'pie' ? 'Values' : 'Y Axis'}
            </label>
            <select
              value={yColumn}
              onChange={(e) => setYColumn(e.target.value)}
              className="w-full px-2 py-1.5 text-sm border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="">Select column...</option>
              {yAxisColumns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
        )}

        {/* Aggregation */}
        {needsAggregation && (
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Aggregation</label>
            <select
              value={aggregation}
              onChange={(e) => setAggregation(e.target.value)}
              className="w-full px-2 py-1.5 text-sm border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {AGGREGATIONS.map(({ value, label }) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>
        )}

        {/* Color/Group By */}
        {chartType !== 'pie' && (
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Color / Group By (optional)</label>
            <select
              value={colorColumn}
              onChange={(e) => setColorColumn(e.target.value)}
              className="w-full px-2 py-1.5 text-sm border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="">None</option>
              {categoricalColumns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
        )}

        {/* Title */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Title (optional)</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Chart title..."
            className="w-full px-2 py-1.5 text-sm border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={loading || !xColumn || (needsXY && chartType !== 'histogram' && !yColumn)}
          className="w-full py-2 bg-primary-500 text-white rounded-lg text-sm font-medium hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Generating...' : 'Generate Chart'}
        </button>

        {error && (
          <p className="text-xs text-red-500 text-center">{error}</p>
        )}
      </div>

      {/* Chart Display */}
      <div className="flex-1 overflow-auto p-3">
        {chart ? (
          <div className="bg-gray-50 rounded-lg p-2">
            <Plot
              data={chart.plotly_json.data}
              layout={{
                ...chart.plotly_json.layout,
                autosize: true,
                margin: { l: 50, r: 30, t: 50, b: 50 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
              }}
              config={{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
              }}
              style={{ width: '100%', height: '300px' }}
              useResizeHandler
            />
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            Select columns and generate a chart
          </div>
        )}
      </div>
    </div>
  );
}
