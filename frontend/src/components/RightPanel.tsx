import { useState } from 'react';
import { Database, BarChart2, Lightbulb, X } from 'lucide-react';
import { useStore } from '../store/useStore';
import { InsightsPanel } from './InsightsPanel';
import { YDataProfilePanel } from './YDataProfilePanel';
import { QuickChartsPanel } from './QuickChartsPanel';

type TabType = 'profile' | 'charts' | 'insights';

export function RightPanel() {
  const { toggleInsightsPanel } = useStore();
  const [activeTab, setActiveTab] = useState<TabType>('profile');

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header with tabs */}
      <div className="p-2 border-b flex items-center justify-between bg-gray-50">
        <div className="flex gap-1 flex-1">
          <button
            onClick={() => setActiveTab('profile')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md transition-colors ${
              activeTab === 'profile'
                ? 'bg-white shadow-sm font-medium text-primary-700 border border-gray-200'
                : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            Profile
          </button>
          <button
            onClick={() => setActiveTab('charts')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md transition-colors ${
              activeTab === 'charts'
                ? 'bg-white shadow-sm font-medium text-primary-700 border border-gray-200'
                : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            Charts
          </button>
          <button
            onClick={() => setActiveTab('insights')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md transition-colors ${
              activeTab === 'insights'
                ? 'bg-white shadow-sm font-medium text-primary-700 border border-gray-200'
                : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
            }`}
          >
            <Lightbulb className="w-3.5 h-3.5" />
            Insights
          </button>
        </div>
        <button
          onClick={toggleInsightsPanel}
          className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'profile' && <YDataProfilePanel />}
        {activeTab === 'charts' && <QuickChartsPanel />}
        {activeTab === 'insights' && <InsightsPanel />}
      </div>
    </div>
  );
}
