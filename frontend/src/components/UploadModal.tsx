import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { X, Upload, FileSpreadsheet, AlertCircle, Check, Layers } from 'lucide-react';
import { useStore } from '../store/useStore';
import type { Dataset } from '../store/useStore';
import { api } from '../services/api';

interface UploadModalProps {
  onClose: () => void;
}

function isExcelFile(file: File): boolean {
  const ext = file.name.split('.').pop()?.toLowerCase();
  return ext === 'xlsx' || ext === 'xls';
}

interface SheetInfo {
  name: string;
  row_count: number;
  column_count?: number;
}

export function UploadModal({ onClose }: UploadModalProps) {
  const { setSessionId, setActiveDataset, setDatasets, setPreviewData } = useStore();
  
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sheets, setSheets] = useState<SheetInfo[] | null>(null);
  const [loadingSheets, setLoadingSheets] = useState(false);
  const [selectedSheets, setSelectedSheets] = useState<Set<string>>(new Set());
  const [options, setOptions] = useState({
    sampleMode: false,
    maxRows: undefined as number | undefined,
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
      setSheets(null);
      setSelectedSheets(new Set());
    }
  }, []);

  useEffect(() => {
    if (!file || !isExcelFile(file)) return;
    let cancelled = false;
    setLoadingSheets(true);
    api.getExcelSheets(file)
      .then((data: { sheets: SheetInfo[] }) => {
        if (cancelled) return;
        setSheets(data.sheets || []);
        setSelectedSheets(new Set((data.sheets || []).map((s: SheetInfo) => s.name)));
      })
      .catch(() => {
        if (!cancelled) setSheets([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingSheets(false);
      });
    return () => { cancelled = true; };
  }, [file]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024, // 500MB
  });

  const toggleSheet = (name: string) => {
    setSelectedSheets((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const selectAllSheets = () => {
    if (!sheets) return;
    setSelectedSheets(new Set(sheets.map((s) => s.name)));
  };

  const handleUpload = async () => {
    if (!file) return;

    const isExcel = isExcelFile(file);
    const useMultiSheet = isExcel && sheets && sheets.length > 0 && selectedSheets.size > 0;

    setUploading(true);
    setError(null);

    try {
      if (useMultiSheet) {
        const result = await api.uploadExcelMulti(file, Array.from(selectedSheets), options);
        if (result.session_id) setSessionId(result.session_id);
        const datasets = (result.datasets || []).map((d: any) => ({
          name: d.name,
          filename: d.filename,
          rowCount: d.row_count,
          columnCount: d.column_count,
          columns: d.columns,
          inferredTypes: d.inferred_types || {},
          dateColumns: d.date_columns || [],
          numericColumns: d.numeric_columns || [],
          categoricalColumns: d.categorical_columns || [],
          isSampled: d.is_sampled,
          fullRowCount: d.full_row_count,
        }));
        setDatasets(datasets);
        const first = datasets.find((d: Dataset) => d.name === result.active_dataset_name) || datasets[0];
        setActiveDataset(first || null);
        setPreviewData(result.preview?.rows || []);
      } else {
        const result = await api.uploadFile(file, options);
        if (result.session_id) setSessionId(result.session_id);
        const dataset = {
          name: result.dataset.name,
          filename: result.dataset.filename,
          rowCount: result.dataset.row_count,
          columnCount: result.dataset.column_count,
          columns: result.dataset.columns,
          inferredTypes: result.dataset.inferred_types,
          dateColumns: result.dataset.date_columns,
          numericColumns: result.dataset.numeric_columns,
          categoricalColumns: result.dataset.categorical_columns,
          isSampled: result.dataset.is_sampled,
          fullRowCount: result.dataset.full_row_count,
        };
        setDatasets([dataset]);
        setActiveDataset(dataset);
        setPreviewData(result.preview?.rows || []);
      }
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg">
        {/* Header */}
        <div className="p-4 border-b flex items-center justify-between">
          <h2 className="text-lg font-semibold">Upload Data File</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-lg"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-primary-500 bg-primary-50'
                : file
                ? 'border-green-500 bg-green-50'
                : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
            }`}
          >
            <input {...getInputProps()} />
            
            {file ? (
              <div className="flex items-center justify-center gap-3">
                <div className="w-12 h-12 rounded-lg bg-green-100 flex items-center justify-center">
                  <FileSpreadsheet className="w-6 h-6 text-green-600" />
                </div>
                <div className="text-left">
                  <p className="font-medium text-gray-900">{file.name}</p>
                  <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
                </div>
                <Check className="w-5 h-5 text-green-600" />
              </div>
            ) : (
              <>
                <div className="w-12 h-12 rounded-xl bg-gray-100 flex items-center justify-center mx-auto mb-3">
                  <Upload className="w-6 h-6 text-gray-400" />
                </div>
                <p className="text-gray-600 mb-1">
                  {isDragActive ? 'Drop your file here' : 'Drag & drop your file here'}
                </p>
                <p className="text-sm text-gray-400">or click to browse</p>
              </>
            )}
          </div>

          {/* Supported formats */}
          <p className="text-xs text-gray-400 mt-3 text-center">
            Supports CSV, TSV, XLSX, XLS • Max 500MB
          </p>

          {/* CSV/TSV: single table – explain why no sheet picker */}
          {file && !isExcelFile(file) && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Single table.</strong> CSV/TSV files have no sheets. To load multiple sheets (tabs), upload the original <strong>Excel file (.xlsx or .xls)</strong> and choose which sheets to load.
              </p>
            </div>
          )}

          {/* Excel: sheet picker (tabs) – always show when Excel file selected */}
          {file && isExcelFile(file) && (
            <div className="mt-4 p-3 bg-primary-50 border border-primary-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-primary-800 flex items-center gap-1">
                  <Layers className="w-4 h-4" />
                  Sheets (tabs) to load
                </span>
                {sheets && sheets.length > 1 && (
                  <button
                    type="button"
                    onClick={selectAllSheets}
                    className="text-xs text-primary-600 hover:underline font-medium"
                  >
                    Select all
                  </button>
                )}
              </div>
              {loadingSheets ? (
                <p className="text-sm text-primary-700">Loading sheet list...</p>
              ) : sheets && sheets.length > 0 ? (
                <div className="max-h-40 overflow-y-auto space-y-1">
                  {sheets.map((sheet) => (
                    <label key={sheet.name} className="flex items-center gap-2 py-1.5 cursor-pointer hover:bg-white/50 rounded px-1">
                      <input
                        type="checkbox"
                        checked={selectedSheets.has(sheet.name)}
                        onChange={() => toggleSheet(sheet.name)}
                        className="w-4 h-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                      />
                      <span className="text-sm text-gray-800 truncate">{sheet.name}</span>
                      {sheet.row_count != null && (
                        <span className="text-xs text-gray-500">({sheet.row_count.toLocaleString()} rows)</span>
                      )}
                    </label>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-amber-700">
                  Could not read sheets. Make sure the file is a valid .xlsx or .xls Excel file.
                </p>
              )}
            </div>
          )}

          {/* Options */}
          {file && (
            <div className="mt-4 space-y-3">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={options.sampleMode}
                  onChange={(e) => setOptions({ ...options, sampleMode: e.target.checked })}
                  className="w-4 h-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-gray-700">
                  Preview mode (load sample for large files)
                </span>
              </label>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2 text-sm text-red-700">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t flex justify-end gap-3">
          <button
            onClick={onClose}
            className="btn btn-secondary"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={Boolean(
              !file ||
              uploading ||
              loadingSheets ||
              (file && isExcelFile(file) && sheets && sheets.length > 0 && selectedSheets.size === 0)
            )}
            className="btn btn-primary"
          >
            {uploading ? (
              <>
                <div className="w-4 h-4 spinner mr-2" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Upload & Analyze
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
