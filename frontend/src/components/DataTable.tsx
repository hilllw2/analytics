import { useMemo, useState } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  createColumnHelper,
  SortingState,
} from '@tanstack/react-table';
import { ChevronUp, ChevronDown, Download, Search } from 'lucide-react';

interface DataTableProps {
  data: {
    columns?: string[];
    data?: any[];
    dtypes?: Record<string, string>;
    // For series
    name?: string;
    index?: string[];
    values?: any[];
  } | any;
  maxRows?: number;
}

export function DataTable({ data, maxRows = 100 }: DataTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState('');

  // Handle different data formats
  const { columns, rows } = useMemo(() => {
    // DataFrame format
    if (data?.columns && data?.data) {
      return {
        columns: data.columns,
        rows: data.data.slice(0, maxRows),
      };
    }
    
    // Series format
    if (data?.index && data?.values) {
      return {
        columns: ['Index', data.name || 'Value'],
        rows: data.index.map((idx: string, i: number) => ({
          Index: idx,
          [data.name || 'Value']: data.values[i],
        })).slice(0, maxRows),
      };
    }
    
    // Plain array of objects
    if (Array.isArray(data) && data.length > 0) {
      return {
        columns: Object.keys(data[0]),
        rows: data.slice(0, maxRows),
      };
    }
    
    // Single value or unknown format
    return { columns: [], rows: [] };
  }, [data, maxRows]);

  // Create column definitions
  const columnHelper = createColumnHelper<any>();
  const tableColumns = useMemo(() => 
    columns.map((col: string) => 
      columnHelper.accessor(col, {
        header: col,
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined || value === '') {
            return <span className="text-gray-300">—</span>;
          }
          if (typeof value === 'number') {
            return <span className="font-mono">{value.toLocaleString()}</span>;
          }
          return String(value);
        },
      })
    ),
    [columns, columnHelper]
  );

  const table = useReactTable({
    data: rows,
    columns: tableColumns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  if (columns.length === 0 || rows.length === 0) {
    // Render single value or message
    if (typeof data === 'number' || typeof data === 'string') {
      return (
        <div className="p-4 text-center">
          <span className="text-2xl font-semibold text-gray-900">
            {typeof data === 'number' ? data.toLocaleString() : data}
          </span>
        </div>
      );
    }
    return (
      <div className="p-4 text-center text-gray-500">
        No data to display
      </div>
    );
  }

  const handleDownload = () => {
    const csvContent = [
      columns.join(','),
      ...rows.map((row: any) => 
        columns.map((col: string) => {
          const val = row[col];
          if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
            return `"${val.replace(/"/g, '""')}"`;
          }
          return val ?? '';
        }).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'data_export.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="overflow-hidden">
      {/* Header */}
      <div className="p-3 border-b flex items-center justify-between bg-gray-50">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">
            {rows.length.toLocaleString()} rows × {columns.length} columns
          </span>
          {data?.data?.length > maxRows && (
            <span className="text-xs text-amber-600">
              (showing first {maxRows})
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-2 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              value={globalFilter}
              onChange={(e) => setGlobalFilter(e.target.value)}
              placeholder="Search..."
              className="pl-8 pr-3 py-1 text-sm border rounded-lg w-40 focus:outline-none focus:ring-1 focus:ring-primary-500"
            />
          </div>
          <button
            onClick={handleDownload}
            className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
            title="Download as CSV"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto max-h-96">
        <table className="data-grid">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className="cursor-pointer select-none"
                  >
                    <div className="flex items-center gap-1">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() && (
                        header.column.getIsSorted() === 'asc' 
                          ? <ChevronUp className="w-3 h-3" />
                          : <ChevronDown className="w-3 h-3" />
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
