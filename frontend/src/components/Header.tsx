import { 
  Upload, 
  Download, 
  Database, 
  Lightbulb,
  BarChart3,
  LogOut
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { api } from '../services/api';

interface HeaderProps {
  onUpload: () => void;
}

export function Header({ onUpload }: HeaderProps) {
  const { 
    activeDataset, 
    toggleDataPanel, 
    toggleInsightsPanel,
    showDataPanel,
    showInsightsPanel,
    clearSession
  } = useStore();

  const handleExportBundle = async () => {
    try {
      const blob = await api.downloadBundle();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'datachat_export.zip';
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const handleEndSession = async () => {
    if (confirm('End session? All data will be permanently deleted.')) {
      try {
        await api.endSession();
        clearSession();
        window.location.reload();
      } catch (error) {
        console.error('Failed to end session:', error);
      }
    }
  };

  return (
    <header className="h-14 bg-white border-b flex items-center justify-between px-4">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-600 to-accent-500 flex items-center justify-center">
          <BarChart3 className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="font-semibold text-gray-900">DataChat</h1>
          <p className="text-xs text-gray-500">Natural Language Analytics</p>
        </div>
      </div>

      {/* Center: Dataset Info */}
      {activeDataset && (
        <div className="flex items-center gap-2 text-sm">
          <Database className="w-4 h-4 text-gray-400" />
          <span className="font-medium">{activeDataset.name}</span>
          <span className="text-gray-400">•</span>
          <span className="text-gray-500">
            {activeDataset.rowCount.toLocaleString()} rows × {activeDataset.columnCount} columns
          </span>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2">
        {/* Toggle panels */}
        <button
          onClick={toggleDataPanel}
          className={`btn btn-ghost p-2 ${showDataPanel ? 'bg-primary-50 text-primary-600' : ''}`}
          title="Toggle Data Panel"
        >
          <Database className="w-5 h-5" />
        </button>
        
        <button
          onClick={toggleInsightsPanel}
          className={`btn btn-ghost p-2 ${showInsightsPanel ? 'bg-primary-50 text-primary-600' : ''}`}
          title="Toggle Insights Panel"
        >
          <Lightbulb className="w-5 h-5" />
        </button>

        <div className="w-px h-6 bg-gray-200 mx-1" />

        {/* Upload */}
        <button
          onClick={onUpload}
          className="btn btn-secondary flex items-center gap-2"
        >
          <Upload className="w-4 h-4" />
          Upload
        </button>

        {/* Export */}
        {activeDataset && (
          <button
            onClick={handleExportBundle}
            className="btn btn-secondary flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        )}

        {/* End Session */}
        <button
          onClick={handleEndSession}
          className="btn btn-ghost p-2 text-red-500 hover:bg-red-50"
          title="End Session"
        >
          <LogOut className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}
