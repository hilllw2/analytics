import { useState, useEffect } from 'react';
import { Search, Filter, ArrowUpDown, RefreshCw } from 'lucide-react';
import { useStore } from '../store/useStore';
import { api } from '../services/api';
import { DataTable } from './DataTable';

export function DataPanel() {
  const { activeDataset, previewData, setPreviewData } = useStore();
  
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [sortColumn, setSortColumn] = useState<string | undefined>();
  const [sortAsc, setSortAsc] = useState(true);
  const [offset, setOffset] = useState(0);
  const [totalRows, setTotalRows] = useState(0);
  const limit = 100;

  // Load preview data
  useEffect(() => {
    const loadPreview = async () => {
      if (!activeDataset) return;
      
      setLoading(true);
      try {
        const data = await api.getPreview({
          offset,
          limit,
          sortColumn,
          sortAscending: sortAsc,
          search: search || undefined,
        });
        
        setPreviewData(data.rows);
        setTotalRows(data.total_rows);
      } catch (error) {
        console.error('Failed to load preview:', error);
      } finally {
        setLoading(false);
      }
    };

    loadPreview();
  }, [activeDataset, offset, sortColumn, sortAsc, search, setPreviewData]);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortAsc(!sortAsc);
    } else {
      setSortColumn(column);
      setSortAsc(true);
    }
    setOffset(0);
  };

  const handleSearch = (value: string) => {
    setSearch(value);
    setOffset(0);
  };

  if (!activeDataset) return null;

  const columns = activeDataset.columns;
  const totalPages = Math.ceil(totalRows / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-gray-900">Data Preview</h2>
          <span className="text-sm text-gray-500">
            {totalRows.toLocaleString()} rows
          </span>
        </div>
        
        {/* Search */}
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search across all columns..."
            className="input pl-9 py-2"
          />
        </div>
      </div>

      {/* Column Types Bar */}
      <div className="px-4 py-2 border-b bg-gray-50 flex gap-4 text-xs">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-blue-400" />
          Numeric ({activeDataset.numericColumns.length})
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-green-400" />
          Date ({activeDataset.dateColumns.length})
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-gray-400" />
          Text ({activeDataset.categoricalColumns.length})
        </span>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <div className="w-8 h-8 spinner" />
          </div>
        ) : (
          <table className="data-grid">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th
                    key={col}
                    onClick={() => handleSort(col)}
                    className="cursor-pointer hover:bg-gray-100"
                  >
                    <div className="flex items-center gap-1">
                      <span className={`w-2 h-2 rounded-full ${
                        activeDataset.numericColumns.includes(col) ? 'bg-blue-400' :
                        activeDataset.dateColumns.includes(col) ? 'bg-green-400' :
                        'bg-gray-300'
                      }`} />
                      <span className="truncate max-w-[150px]" title={col}>{col}</span>
                      {sortColumn === col && (
                        <ArrowUpDown className="w-3 h-3" />
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {previewData.map((row, i) => (
                <tr key={i}>
                  {columns.map((col) => (
                    <td key={col} className="max-w-[200px] truncate" title={String(row[col] ?? '')}>
                      {row[col] === null || row[col] === undefined || row[col] === '' ? (
                        <span className="text-gray-300">—</span>
                      ) : typeof row[col] === 'number' ? (
                        <span className="font-mono text-right">{row[col].toLocaleString()}</span>
                      ) : (
                        String(row[col])
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      <div className="p-3 border-t flex items-center justify-between bg-gray-50">
        <span className="text-sm text-gray-500">
          Showing {offset + 1}-{Math.min(offset + limit, totalRows)} of {totalRows.toLocaleString()}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => setOffset(Math.max(0, offset - limit))}
            disabled={offset === 0}
            className="btn btn-secondary text-sm py-1"
          >
            Previous
          </button>
          <span className="py-1 px-2 text-sm">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setOffset(offset + limit)}
            disabled={offset + limit >= totalRows}
            className="btn btn-secondary text-sm py-1"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
