import { useState, useEffect, useRef } from 'react';
import { RefreshCw, Maximize2, Minimize2 } from 'lucide-react';
import { useStore } from '../store/useStore';
import { api } from '../services/api';

export function YDataProfilePanel() {
  const { activeDataset } = useStore();
  const [loading, setLoading] = useState(false);
  const [htmlReport, setHtmlReport] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const loadProfile = async () => {
    if (!activeDataset) return;
    
    setLoading(true);
    setError(null);
    try {
      const data = await api.getYDataProfile(true);
      setHtmlReport(data.html);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate profile');
      console.error('Failed to load ydata profile:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Auto-load when dataset changes
    if (activeDataset && !htmlReport) {
      loadProfile();
    }
  }, [activeDataset]);

  useEffect(() => {
    // Write HTML to iframe
    if (iframeRef.current && htmlReport) {
      const doc = iframeRef.current.contentDocument;
      if (doc) {
        doc.open();
        doc.write(htmlReport);
        doc.close();
      }
    }
  }, [htmlReport]);

  if (!activeDataset) return null;

  return (
    <div className={`flex flex-col h-full bg-white ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Header */}
      <div className="p-3 border-b flex items-center justify-between bg-white">
        <span className="text-sm font-medium text-gray-700">YData Profiling Report</span>
        <div className="flex gap-1">
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
          <button
            onClick={loadProfile}
            disabled={loading}
            className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
            title="Refresh profile"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <div className="w-8 h-8 spinner mb-3" />
            <p className="text-sm">Generating data profile...</p>
            <p className="text-xs text-gray-400 mt-1">This may take a moment for large datasets</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full text-red-500 p-4">
            <p className="text-sm text-center">{error}</p>
            <button
              onClick={loadProfile}
              className="mt-3 px-4 py-2 bg-primary-500 text-white rounded-lg text-sm hover:bg-primary-600"
            >
              Retry
            </button>
          </div>
        ) : htmlReport ? (
          <iframe
            ref={iframeRef}
            className="w-full h-full border-0"
            title="YData Profile Report"
            sandbox="allow-scripts allow-same-origin"
          />
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 p-4">
            <p className="text-sm text-center mb-3">Click to generate a comprehensive data profile report</p>
            <button
              onClick={loadProfile}
              className="px-4 py-2 bg-primary-500 text-white rounded-lg text-sm hover:bg-primary-600"
            >
              Generate Profile
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
