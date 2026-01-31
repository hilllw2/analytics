import { 
  Upload, 
  FileSpreadsheet, 
  TrendingUp, 
  PieChart, 
  Users,
  AlertTriangle,
  Layers,
  BarChart2,
  Target
} from 'lucide-react';
import { useStore } from '../store/useStore';

interface SidebarProps {
  onUpload: () => void;
}

export function Sidebar({ onUpload }: SidebarProps) {
  const { activeDataset, datasets } = useStore();

  const quickActions = [
    { icon: TrendingUp, label: 'KPI Analysis', query: 'Show me the key metrics' },
    { icon: PieChart, label: 'Distribution', query: 'What is the distribution of the main metric?' },
    { icon: Users, label: 'Segments', query: 'Compare segments by the main metric' },
    { icon: AlertTriangle, label: 'Anomalies', query: 'Find anomalies in the data' },
    { icon: Layers, label: 'Correlations', query: 'Show correlations between numeric columns' },
    { icon: Target, label: 'Top Drivers', query: 'What are the main drivers?' },
  ];

  return (
    <aside className="w-56 bg-white border-r flex flex-col">
      {/* Datasets */}
      <div className="p-4 border-b">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
          Datasets
        </h3>
        
        {datasets.length === 0 ? (
          <button
            onClick={onUpload}
            className="w-full p-3 border-2 border-dashed border-gray-200 rounded-lg text-center hover:border-primary-300 hover:bg-primary-50 transition-colors"
          >
            <Upload className="w-6 h-6 text-gray-400 mx-auto mb-1" />
            <span className="text-sm text-gray-500">Upload file</span>
          </button>
        ) : (
          <div className="space-y-1">
            {datasets.map((ds) => (
              <div
                key={ds.name}
                className={`p-2 rounded-lg flex items-center gap-2 cursor-pointer transition-colors ${
                  activeDataset?.name === ds.name
                    ? 'bg-primary-50 text-primary-700'
                    : 'hover:bg-gray-50'
                }`}
              >
                <FileSpreadsheet className="w-4 h-4" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{ds.name}</p>
                  <p className="text-xs text-gray-500">
                    {ds.rowCount.toLocaleString()} rows
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      {activeDataset && (
        <div className="p-4 flex-1 overflow-y-auto">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            Quick Analysis
          </h3>
          
          <div className="space-y-1">
            {quickActions.map((action) => (
              <button
                key={action.label}
                onClick={() => {
                  // Trigger the query in chat
                  const chatInput = document.querySelector('textarea');
                  if (chatInput) {
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                      window.HTMLTextAreaElement.prototype,
                      'value'
                    )?.set;
                    nativeInputValueSetter?.call(chatInput, action.query);
                    chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                  }
                }}
                className="w-full p-2 text-left rounded-lg flex items-center gap-2 hover:bg-gray-50 transition-colors text-sm text-gray-600 hover:text-gray-900"
              >
                <action.icon className="w-4 h-4 text-gray-400" />
                {action.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Column Quick Reference */}
      {activeDataset && (
        <div className="p-4 border-t max-h-48 overflow-y-auto">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Columns
          </h3>
          <div className="space-y-1">
            {activeDataset.columns.slice(0, 10).map((col) => (
              <div
                key={col}
                className="text-xs text-gray-600 truncate"
                title={col}
              >
                <span className={`inline-block w-2 h-2 rounded-full mr-1 ${
                  activeDataset.numericColumns.includes(col) ? 'bg-blue-400' :
                  activeDataset.dateColumns.includes(col) ? 'bg-green-400' :
                  'bg-gray-300'
                }`} />
                {col}
              </div>
            ))}
            {activeDataset.columns.length > 10 && (
              <div className="text-xs text-gray-400">
                +{activeDataset.columns.length - 10} more
              </div>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}
