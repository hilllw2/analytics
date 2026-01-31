"""
Upload API Routes
Handles file upload, parsing, and data ingestion.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Any
import json
import pandas as pd
import numpy as np
from datetime import datetime, date

from app.core.session_manager import session_manager
from app.core.config import settings
from app.services.data_ingestion import data_ingestion
from app.services.data_profiler import data_profiler
from app.services.auto_insights import auto_insights


router = APIRouter()


def make_json_serializable(obj: Any) -> Any:
    """Convert to JSON-serializable format; preserve inf/NaN as strings so data is not lost."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return "NaN"
        if np.isposinf(obj):
            return "Infinity"
        if np.isneginf(obj):
            return "-Infinity"
        return float(obj)
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat() if pd.notna(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [make_json_serializable(x) for x in obj.tolist()]
    if pd.isna(obj):
        return "NaN"
    return str(obj)


def serialize_dataframe_rows(df: pd.DataFrame) -> List[dict]:
    """Convert DataFrame rows to JSON-serializable list of dicts."""
    rows = []
    for _, row in df.iterrows():
        rows.append({col: make_json_serializable(val) for col, val in row.items()})
    return rows


@router.post("/file")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    sample_mode: bool = Query(False, description="Preview mode with sampled data"),
    max_rows: Optional[int] = Query(None, description="Max rows to load"),
    sheet_name: Optional[str] = Query(None, description="Sheet name for Excel files"),
    clean_pivot: bool = Query(True, description="Run cleaning pipeline: drop all-NaN rows/columns (recommended for pivot tables)"),
    drop_na_rows: str = Query("all", description="Drop rows: 'all' = only all-NaN rows, 'any' = rows with any NaN")
):
    """
    Upload a data file (CSV, TSV, XLSX, XLS).
    
    Cleaning pipeline (when clean_pivot=True): drops columns that are entirely NaN,
    and rows that are entirely NaN (or rows with any NaN if drop_na_rows='any').
    Creates a new session if no session_id provided.
    """
    # Validate file type
    allowed_extensions = {'.csv', '.tsv', '.xlsx', '.xls'}
    file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Maximum: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Get or create session
    if session_id:
        session = session_manager.get_session(session_id)
        if not session:
            # Session expired or invalid, create new one
            session = session_manager.create_session()
    else:
        session = session_manager.create_session()
    
    try:
        # Load the file (cleaning pipeline runs by default for pivot-table uploads)
        df, metadata = await data_ingestion.load_file(
            content=content,
            filename=file.filename,
            sheet_name=sheet_name,
            sample_mode=sample_mode,
            max_rows=max_rows,
            clean_pivot=clean_pivot,
            drop_na_rows=drop_na_rows if drop_na_rows in ("all", "any") else "all"
        )
        
        # Create dataset info
        dataset_name = file.filename.rsplit('.', 1)[0]
        dataset_info = data_ingestion.create_dataset_info(dataset_name, df, metadata)
        
        # Add to session
        session.datasets[dataset_name] = dataset_info
        session.active_dataset_name = dataset_name
        
        # Generate preview data (properly serialized)
        preview_df = df.head(settings.PREVIEW_ROWS)
        preview_data = serialize_dataframe_rows(preview_df)
        
        return JSONResponse({
            "success": True,
            "session_id": session.id,
            "dataset": {
                "name": dataset_name,
                "filename": file.filename,
                "row_count": dataset_info.row_count,
                "column_count": dataset_info.column_count,
                "columns": list(df.columns),
                "inferred_types": metadata.get("inferred_types", {}),
                "date_columns": metadata.get("date_columns", []),
                "numeric_columns": metadata.get("numeric_columns", []),
                "categorical_columns": metadata.get("categorical_columns", []),
                "is_sampled": metadata.get("is_sampled", False),
                "full_row_count": metadata.get("full_row_count"),
                "warnings": metadata.get("warnings", []),
                "cleaning_applied": metadata.get("cleaning_applied")
            },
            "preview": {
                "rows": preview_data,
                "row_count": len(preview_data)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/excel-sheets")
@router.post("/excel-sheets")
async def get_excel_sheets(
    file: UploadFile = File(...),
):
    """
    Get list of sheets in an Excel file. Use POST when sending file in body.
    """
    content = await file.read()
    
    try:
        sheets = await data_ingestion.get_excel_sheets(content)
        return {"sheets": sheets}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read Excel file: {str(e)}")


@router.post("/file-excel-multi")
async def upload_excel_multi_sheets(
    file: UploadFile = File(...),
    sheet_names: str = Form(..., description='JSON array of sheet names, e.g. ["Sheet1", "Sheet2"]'),
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    sample_mode: bool = Query(False),
    max_rows: Optional[int] = Query(None),
    clean_pivot: bool = Query(True, description="Run cleaning pipeline (drop all-NaN rows/columns)"),
    drop_na_rows: str = Query("all", description="'all' = drop only all-NaN rows, 'any' = drop rows with any NaN"),
):
    """
    Upload an Excel file and load multiple sheets as separate datasets (tabs).
    Each sheet becomes a dataset named "FileName - SheetName".
    """
    import json
    allowed = {'.xlsx', '.xls'}
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only .xlsx and .xls files are supported for multi-sheet upload")
    
    try:
        names_list = json.loads(sheet_names)
        if not names_list or not isinstance(names_list, list):
            raise HTTPException(status_code=400, detail="sheet_names must be a non-empty JSON array of strings")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid sheet_names JSON: {e}")
    
    content = await file.read()
    base_name = file.filename.rsplit('.', 1)[0]
    
    if session_id:
        session = session_manager.get_session(session_id)
        if not session:
            session = session_manager.create_session()
    else:
        session = session_manager.create_session()
    
    datasets_out = []
    first_preview = None
    first_name = None
    
    for sheet_name in names_list:
        if not isinstance(sheet_name, str):
            continue
        try:
            df, metadata = await data_ingestion.load_file(
                content=content,
                filename=file.filename,
                sheet_name=sheet_name,
                sample_mode=sample_mode,
                max_rows=max_rows,
                clean_pivot=clean_pivot,
                drop_na_rows=drop_na_rows if drop_na_rows in ("all", "any") else "all",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load sheet '{sheet_name}': {str(e)}")
        
        dataset_name = f"{base_name} - {sheet_name}"
        dataset_info = data_ingestion.create_dataset_info(dataset_name, df, metadata)
        session.datasets[dataset_name] = dataset_info
        
        preview_df = df.head(settings.PREVIEW_ROWS)
        preview_data = serialize_dataframe_rows(preview_df)
        
        datasets_out.append({
            "name": dataset_name,
            "filename": file.filename,
            "sheet_name": sheet_name,
            "row_count": dataset_info.row_count,
            "column_count": dataset_info.column_count,
            "columns": list(df.columns),
            "inferred_types": metadata.get("inferred_types", {}),
            "date_columns": metadata.get("date_columns", []),
            "numeric_columns": metadata.get("numeric_columns", []),
            "categorical_columns": metadata.get("categorical_columns", []),
            "is_sampled": metadata.get("is_sampled", False),
            "full_row_count": metadata.get("full_row_count"),
        })
        
        if first_preview is None:
            first_name = dataset_name
            session.active_dataset_name = dataset_name
            first_preview = {"rows": preview_data, "row_count": len(preview_data)}
    
    return JSONResponse({
        "success": True,
        "session_id": session.id,
        "datasets": datasets_out,
        "active_dataset_name": first_name,
        "preview": first_preview,
    })


@router.post("/load-full")
async def load_full_dataset(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Load the full dataset (if previously loaded in sample mode).
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    # Get dataset
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = session.datasets[name]
    
    if not dataset.is_sampled:
        return {"message": "Dataset is already fully loaded"}
    
    # This would require re-uploading the file since we don't store it
    # In a real implementation, you might temporarily cache the file
    return JSONResponse({
        "success": False,
        "message": "Please re-upload the file without sample mode to load full data"
    })


@router.get("/preview")
async def get_preview(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    sort_column: Optional[str] = Query(None),
    sort_ascending: bool = Query(True),
    search: Optional[str] = Query(None),
    search_column: Optional[str] = Query(None)
):
    """
    Get paginated preview of dataset with optional filtering and sorting.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = session.datasets[name]
    df = dataset.df.copy()
    
    # Apply search filter
    if search and search_column and search_column in df.columns:
        df = df[df[search_column].astype(str).str.contains(search, case=False, na=False)]
    elif search:
        # Search across all columns
        mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
        df = df[mask]
    
    # Apply sorting
    if sort_column and sort_column in df.columns:
        df = df.sort_values(sort_column, ascending=sort_ascending)
    
    # Get total after filtering
    total_rows = len(df)
    
    # Apply pagination
    df_page = df.iloc[offset:offset + limit]
    
    return {
        "rows": serialize_dataframe_rows(df_page),
        "total_rows": total_rows,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total_rows
    }


@router.get("/column-values")
async def get_column_values(
    column: str,
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Get unique values for a column (for filter dropdowns).
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = session.datasets[name]
    
    if column not in dataset.df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
    
    raw_values = dataset.df[column].dropna().unique()[:limit]
    values = [make_json_serializable(v) for v in raw_values]
    
    return {
        "column": column,
        "values": values,
        "total_unique": dataset.df[column].nunique()
    }


@router.post("/profile")
async def generate_profile(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None),
    include_correlations: bool = Query(True)
):
    """
    Generate a data health report for the dataset.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = session.datasets[name]
    
    try:
        # Generate health report
        report = await data_profiler.generate_health_report(dataset, include_correlations)
        
        # Cache the report
        session.cached_results[f"profile_{name}"] = report
        
        # Helper to serialize values
        def serialize_value(v):
            if v is None:
                return None
            if isinstance(v, (int, float, str, bool)):
                return v
            if hasattr(v, 'isoformat'):
                return v.isoformat()
            if hasattr(v, 'tolist'):
                return v.tolist()
            return str(v)
        
        def serialize_dict(d):
            if d is None:
                return None
            if isinstance(d, dict):
                return {k: serialize_dict(v) for k, v in d.items()}
            if isinstance(d, list):
                return [serialize_dict(i) for i in d]
            return serialize_value(d)
        
        return {
            "report": {
                "generated_at": report.generated_at,
                "dataset_name": report.dataset_name,
                "row_count": report.row_count,
                "column_count": report.column_count,
                "memory_usage_mb": report.memory_usage_mb,
                "total_missing_cells": report.total_missing_cells,
                "total_missing_pct": report.total_missing_pct,
                "duplicate_row_count": report.duplicate_row_count,
                "duplicate_row_pct": report.duplicate_row_pct,
                "numeric_columns": report.numeric_columns,
                "categorical_columns": report.categorical_columns,
                "datetime_columns": report.datetime_columns,
                "constant_columns": report.constant_columns,
                "high_missing_columns": report.high_missing_columns,
                "column_profiles": serialize_dict(report.column_profiles),
                "correlation_matrix": serialize_dict(report.correlation_matrix),
                "high_correlations": serialize_dict(report.high_correlations),
                "missingness_heatmap": serialize_dict(report.missingness_heatmap),
                "warnings": report.warnings or []
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate profile: {str(e)}")


@router.post("/insights")
async def generate_insights(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None),
    max_insights: int = Query(15, ge=1, le=30)
):
    """
    Generate automatic insights for the dataset.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = session.datasets[name]
    
    try:
        # Generate insights
        insights = await auto_insights.generate_insights(dataset, max_insights)
        
        # Generate suggested questions
        suggestions = await auto_insights.generate_suggested_questions(dataset)
        
        return {
            "insights": insights,
            "suggested_questions": suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.post("/ydata-profile")
async def generate_ydata_profile(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None),
    minimal: bool = Query(True, description="Use minimal mode for faster report")
):
    """
    Generate a ydata-profiling HTML report.
    Returns HTML that can be displayed in an iframe.
    """
    from ydata_profiling import ProfileReport
    import tempfile
    import os
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = session.datasets[name]
    
    try:
        # Sample data if too large
        df = dataset.df
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
        
        # Generate profile report
        profile = ProfileReport(
            df, 
            title=f"Data Profile: {dataset.name}",
            minimal=minimal,
            explorative=not minimal,
            progress_bar=False
        )
        
        # Get HTML and cache in session for download/bundle
        html_report = profile.to_html()
        session.cached_results[f"ydata_profile_{name}"] = html_report
        
        return {"html": html_report}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate ydata profile: {str(e)}")


@router.post("/quick-chart")
async def generate_quick_chart(
    session_id: str = Header(..., alias="X-Session-ID"),
    chart_type: str = Query(..., description="bar, line, pie, scatter, histogram, box"),
    x_column: Optional[str] = Query(None),
    y_column: Optional[str] = Query(None),
    color_column: Optional[str] = Query(None),
    title: Optional[str] = Query(None),
    aggregation: Optional[str] = Query("sum", description="sum, mean, count, min, max")
):
    """
    Generate a quick chart without AI - just select columns and chart type.
    """
    import pandas as pd
    import plotly.express as px
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if not session.active_dataset:
        raise HTTPException(status_code=404, detail="No active dataset")
    
    df = session.active_df.copy()
    available = set(df.columns)
    
    def ensure_numeric(ser):
        return pd.to_numeric(ser, errors="coerce")
    
    def validate_col(name, required=True):
        if not name and not required:
            return
        if name not in available:
            raise HTTPException(status_code=400, detail=f"Column '{name}' not found. Available: {list(available)}")
    
    validate_col(x_column, required=(chart_type != "histogram"))
    validate_col(y_column, required=(chart_type in ("bar", "line", "pie", "scatter")))
    if color_column:
        validate_col(color_column, required=False)
    
    # Insights-style color palette (matches visualization service)
    insights_colors = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899", "#84cc16"]
    
    try:
        fig = None
        chart_title = title or f"{chart_type.capitalize()} Chart"
        agg_map = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}
        agg_func = agg_map.get(aggregation, "sum")
        
        if chart_type == "bar":
            if x_column and y_column:
                df[y_column] = ensure_numeric(df[y_column])
                group_cols = [x_column]
                if color_column:
                    group_cols.append(color_column)
                agg_df = df.groupby(group_cols, dropna=False)[y_column].agg(agg_func).reset_index()
                fig = px.bar(agg_df, x=x_column, y=y_column, color=color_column if color_column else None, title=chart_title, color_discrete_sequence=insights_colors)
            else:
                raise HTTPException(status_code=400, detail="Bar chart requires x and y columns")
                
        elif chart_type == "line":
            if x_column and y_column:
                df[y_column] = ensure_numeric(df[y_column])
                line_kw = {"x": x_column, "y": y_column, "title": chart_title, "color_discrete_sequence": insights_colors}
                if color_column:
                    line_kw["color"] = color_column
                fig = px.line(df, **line_kw)
            else:
                raise HTTPException(status_code=400, detail="Line chart requires x and y columns")
                
        elif chart_type == "pie":
            if x_column and y_column:
                df[y_column] = ensure_numeric(df[y_column])
                agg_df = df.groupby(x_column, dropna=False)[y_column].agg(agg_func).reset_index()
                fig = px.pie(agg_df, names=x_column, values=y_column, title=chart_title, color_discrete_sequence=insights_colors)
            else:
                raise HTTPException(status_code=400, detail="Pie chart requires names (x) and values (y) columns")
                
        elif chart_type == "scatter":
            if x_column and y_column:
                df[x_column] = ensure_numeric(df[x_column])
                df[y_column] = ensure_numeric(df[y_column])
                scatter_kw = {"x": x_column, "y": y_column, "title": chart_title, "color_discrete_sequence": insights_colors}
                if color_column:
                    scatter_kw["color"] = color_column
                fig = px.scatter(df, **scatter_kw)
            else:
                raise HTTPException(status_code=400, detail="Scatter chart requires x and y columns")
                
        elif chart_type == "histogram":
            col = x_column or y_column
            if col:
                df[col] = ensure_numeric(df[col])
                hist_kw = {"x": col, "title": chart_title, "color_discrete_sequence": insights_colors}
                if color_column:
                    hist_kw["color"] = color_column
                fig = px.histogram(df.dropna(subset=[col]), **hist_kw)
            else:
                raise HTTPException(status_code=400, detail="Histogram requires a column")
                
        elif chart_type == "box":
            col = y_column or x_column
            if col:
                df[col] = ensure_numeric(df[col])
                box_kw = {"y": col, "title": chart_title, "color_discrete_sequence": insights_colors}
                if color_column:
                    box_kw["x"] = color_column
                fig = px.box(df.dropna(subset=[col]), **box_kw)
            else:
                raise HTTPException(status_code=400, detail="Box plot requires a column")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown chart type: {chart_type}")
        
        if fig is None:
            raise HTTPException(status_code=500, detail="Chart could not be created")
        
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, system-ui, sans-serif", size=12),
            margin=dict(l=60, r=40, t=60, b=50),
            paper_bgcolor="#fafafa",
            plot_bgcolor="#ffffff"
        )
        
        # Use to_json() + json.loads() so the result is JSON-serializable (no numpy types)
        plotly_dict = json.loads(fig.to_json())
        return {
            "chart": {
                "type": chart_type,
                "title": chart_title,
                "plotly_json": plotly_dict
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate chart: {str(e)}")
