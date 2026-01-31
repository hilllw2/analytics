import { useState } from 'react';
import { useStore } from '../store/useStore';
import { api } from '../services/api';

/**
 * Horizontal tab bar for switching between sheets (datasets).
 * Shown when there is at least one dataset; each tab is one sheet.
 */
export function SheetTabs() {
  const { datasets, activeDataset, setActiveDataset, setPreviewData } = useStore();
  const [switching, setSwitching] = useState(false);

  if (!datasets.length || !activeDataset) return null;

  const switchTo = async (ds: (typeof datasets)[0]) => {
    if (activeDataset.name === ds.name) return;
    setActiveDataset(ds);
    setSwitching(true);
    try {
      await api.setActiveDataset(ds.name);
      const preview = await api.getPreview({ limit: 100 });
      setPreviewData(preview.rows || []);
    } catch {
      // keep UI in sync
    } finally {
      setSwitching(false);
    }
  };

  // Display label: "FileName - Sheet1" -> "Sheet1", or full name
  const tabLabel = (name: string) => {
    const sep = ' - ';
    if (name.includes(sep)) return name.split(sep).slice(-1)[0];
    return name;
  };

  return (
    <div className="bg-white border-b px-4 py-2 flex items-center gap-1 overflow-x-auto">
      <span className="text-xs font-medium text-gray-500 uppercase tracking-wider mr-2 shrink-0">
        Sheets:
      </span>
      <div className="flex gap-1 min-w-0">
        {datasets.map((ds) => (
          <button
            key={ds.name}
            type="button"
            onClick={() => switchTo(ds)}
            disabled={switching}
            className={`
              px-3 py-1.5 rounded-md text-sm font-medium whitespace-nowrap transition-colors
              ${activeDataset?.name === ds.name
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}
              disabled:opacity-60
            `}
          >
            {tabLabel(ds.name)}
          </button>
        ))}
      </div>
    </div>
  );
}
