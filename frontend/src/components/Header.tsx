import { useState, useRef, useEffect } from 'react';
import { 
  Upload, 
  Download, 
  Database, 
  Lightbulb,
  BarChart3,
  LogOut,
  ChevronDown,
  FileText,
  FileJson,
  PieChart
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
  const [exportOpen, setExportOpen] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) setExportOpen(false);
    };
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, []);

  const handleExportBundle = async () => {
    setExportOpen(false);
    try {
      const blob = await api.downloadBundle();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `datachat_export_${new Date().toISOString().slice(0, 10)}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const handleExportReport = async () => {
    setExportOpen(false);
    try {
      const blob = await api.generateReport({ format: 'html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'analysis_report.html';
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Report export failed:', error);
    }
  };

  const handleExportYdataProfile = async () => {
    setExportOpen(false);
    try {
      const blob = await api.downloadYdataProfile(activeDataset?.name);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ydata_profile_${activeDataset?.name?.replace(/\s+/g, '_') || 'dataset'}.html`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Generate YData profile first (Upload > YData Profile)';
      alert(msg);
    }
  };

  const handleExportDataQuality = async () => {
    setExportOpen(false);
    try {
      const data = await api.downloadInsights(activeDataset?.name);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'insights_data_quality.json';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Run data profiling first (Upload > Profile)';
      alert(msg);
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

        {/* Export dropdown */}
        {activeDataset && (
          <div className="relative" ref={exportRef}>
            <button
              onClick={() => setExportOpen(!exportOpen)}
              className="btn btn-secondary flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Export
              <ChevronDown className="w-4 h-4" />
            </button>
            {exportOpen && (
              <div className="absolute right-0 top-full mt-1 py-1 w-56 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
                <button
                  onClick={handleExportBundle}
                  className="w-full flex items-center gap-2 px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50"
                >
                  <Download className="w-4 h-4" />
                  Full bundle (ZIP)
                </button>
                <button
                  onClick={handleExportReport}
                  className="w-full flex items-center gap-2 px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50"
                >
                  <FileText className="w-4 h-4" />
                  Report (HTML, with charts)
                </button>
                <button
                  onClick={handleExportYdataProfile}
                  className="w-full flex items-center gap-2 px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50"
                >
                  <PieChart className="w-4 h-4" />
                  YData profile (HTML)
                </button>
                <button
                  onClick={handleExportDataQuality}
                  className="w-full flex items-center gap-2 px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50"
                >
                  <FileJson className="w-4 h-4" />
                  Data quality (JSON)
                </button>
              </div>
            )}
          </div>
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
