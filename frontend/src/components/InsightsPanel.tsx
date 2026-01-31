import { useState, useEffect } from 'react';
import { 
  Lightbulb, 
  TrendingUp, 
  AlertTriangle, 
  BarChart2, 
  RefreshCw,
  ChevronRight,
  Activity
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { api } from '../services/api';

export function InsightsPanel() {
  const { activeDataset, insights, setInsights } = useStore();
  const [loading, setLoading] = useState(false);

  // Load insights when dataset changes
  useEffect(() => {
    const loadInsights = async () => {
      if (!activeDataset) return;
      
      setLoading(true);
      try {
        const insightsData = await api.getInsights(10);
        setInsights(insightsData.insights || []);
      } catch (error) {
        console.error('Failed to load insights:', error);
      } finally {
        setLoading(false);
      }
    };

    loadInsights();
  }, [activeDataset, setInsights]);

  const handleRefresh = async () => {
    if (!activeDataset) return;
    
    setLoading(true);
    try {
      const insightsData = await api.getInsights(10);
      setInsights(insightsData.insights || []);
    } catch (error) {
      console.error('Failed to refresh insights:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'trend': return TrendingUp;
      case 'anomaly': return AlertTriangle;
      case 'segment': return BarChart2;
      case 'distribution': return Activity;
      default: return Lightbulb;
    }
  };

  const getImportanceColor = (importance: string) => {
    switch (importance) {
      case 'high': return 'border-l-red-500';
      case 'medium': return 'border-l-amber-500';
      default: return 'border-l-blue-500';
    }
  };

  if (!activeDataset) return null;

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="p-3 border-b flex items-center justify-between">
        <span className="text-sm text-gray-500">Auto-Generated Insights</span>
        <button
          onClick={handleRefresh}
          disabled={loading}
          className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <div className="w-6 h-6 spinner" />
          </div>
        ) : (
          <div className="space-y-3">
            {insights.length === 0 ? (
              <p className="text-center text-gray-500 text-sm py-4">
                No insights generated yet
              </p>
            ) : (
              insights.map((insight) => {
                const Icon = getCategoryIcon(insight.category);
                return (
                  <div
                    key={insight.id}
                    className={`p-3 bg-gray-50 rounded-lg border-l-4 ${getImportanceColor(insight.importance)}`}
                  >
                    <div className="flex items-start gap-2">
                      <Icon className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-medium text-gray-900 mb-1">
                          {insight.title}
                        </h4>
                        <p className="text-xs text-gray-600 leading-relaxed">
                          {insight.description}
                        </p>
                        {insight.suggestedAction && (
                          <p className="text-xs text-primary-600 mt-2 flex items-center gap-1">
                            <ChevronRight className="w-3 h-3" />
                            {insight.suggestedAction}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
}
