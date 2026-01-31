"""
Export API Routes
Export charts, tables, and reports.
"""

from fastapi import APIRouter, HTTPException, Header, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import io
import json
import zipfile
from datetime import datetime

from app.core.session_manager import session_manager
from app.services.visualization import visualization_service


router = APIRouter()


class ReportRequest(BaseModel):
    """Request for report generation."""
    title: str = "Data Analysis Report"
    include_insights: bool = True
    include_charts: bool = True
    include_tables: bool = True
    format: str = "html"  # 'html', 'markdown'


@router.get("/chart/{chart_id}/png")
async def export_chart_png(
    chart_id: str,
    session_id: str = Header(..., alias="X-Session-ID"),
    width: int = Query(1200, ge=400, le=3000),
    height: int = Query(800, ge=300, le=2000)
):
    """
    Export a chart as PNG image.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if chart_id not in session.charts:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    chart_info = session.charts[chart_id]
    
    try:
        png_bytes = await visualization_service.export_chart_png(chart_info, width, height)
        
        return StreamingResponse(
            io.BytesIO(png_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={chart_info.title.replace(' ', '_')}.png"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PNG export failed: {str(e)}")


@router.get("/chart/{chart_id}/svg")
async def export_chart_svg(
    chart_id: str,
    session_id: str = Header(..., alias="X-Session-ID"),
    width: int = Query(1200, ge=400, le=3000),
    height: int = Query(800, ge=300, le=2000)
):
    """
    Export a chart as SVG image.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if chart_id not in session.charts:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    chart_info = session.charts[chart_id]
    
    try:
        svg_str = await visualization_service.export_chart_svg(chart_info, width, height)
        
        return StreamingResponse(
            io.BytesIO(svg_str.encode('utf-8')),
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": f"attachment; filename={chart_info.title.replace(' ', '_')}.svg"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SVG export failed: {str(e)}")


@router.get("/chart/{chart_id}/data")
async def export_chart_data(
    chart_id: str,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Export underlying chart data as CSV.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if chart_id not in session.charts:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    chart_info = session.charts[chart_id]
    csv_bytes = await visualization_service.export_chart_data(chart_info)
    
    if csv_bytes is None:
        raise HTTPException(status_code=400, detail="No underlying data available for this chart")
    
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={chart_info.title.replace(' ', '_')}_data.csv"
        }
    )


@router.get("/ydata-profile")
async def download_ydata_profile(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Download the YData profiling analysis as an HTML file.
    Generate the profile from Upload > YData Profile first; it will be cached for download.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    html = session.cached_results.get(f"ydata_profile_{name}")
    if not html:
        raise HTTPException(
            status_code=404,
            detail="YData profile not found. Generate it first from Upload > YData Profile, then download."
        )
    
    safe_name = name.replace(" ", "_").replace("/", "-")
    filename = f"ydata_profile_{safe_name}.html"
    return StreamingResponse(
        io.BytesIO(html.encode("utf-8")),
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/insights")
async def download_insights(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Download the data quality / health insights (profile) as JSON.
    Includes missing values, duplicates, warnings, column profiles.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    insights = session.cached_results.get(f"profile_{name}")
    if not insights:
        raise HTTPException(
            status_code=404,
            detail="Insights not found. Run data profiling from Upload first, then download."
        )
    
    # Convert to JSON-serializable dict (handle dataclass/datetime)
    from dataclasses import asdict
    import datetime as dt
    
    if hasattr(insights, "__dataclass_fields__"):
        data = asdict(insights)
    else:
        data = dict(insights) if hasattr(insights, "items") else {"raw": str(insights)}
    
    def _serialize(obj):
        if obj is None or isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, dt.datetime):
            return obj.isoformat()
        if isinstance(obj, float):
            return obj
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(x) for x in obj]
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        try:
            return str(obj)
        except Exception:
            return None
    
    data = _serialize(data)
    return JSONResponse(content=data)


@router.get("/table")
async def export_table_csv(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None),
    max_rows: int = Query(10000, ge=1, le=100000)
):
    """
    Export current dataset or filtered view as CSV.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    name = dataset_name or session.active_dataset_name
    if not name or name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = session.datasets[name]
    df = dataset.df.head(max_rows)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={name}.csv"
        }
    )


@router.post("/report")
async def generate_report(
    request: ReportRequest,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Generate a summary report of the analysis session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if not session.active_dataset:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    dataset = session.active_dataset
    
    # Build report content
    if request.format == 'markdown':
        report = _generate_markdown_report(session, dataset, request)
        media_type = "text/markdown"
        filename = "analysis_report.md"
    else:
        report = _generate_html_report(session, dataset, request)
        media_type = "text/html"
        filename = "analysis_report.html"
    
    return StreamingResponse(
        io.BytesIO(report.encode('utf-8')),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


def _generate_markdown_report(session, dataset, request) -> str:
    """Generate markdown report."""
    lines = [
        f"# {request.title}",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Executive Summary",
        "",
        f"This report analyzes the **{dataset.name}** dataset containing {dataset.row_count:,} rows and {dataset.column_count} columns.",
        "",
        "### Dataset Overview",
        "",
        f"- **Rows**: {dataset.row_count:,}",
        f"- **Columns**: {dataset.column_count}",
        f"- **Numeric Columns**: {', '.join(dataset.numeric_columns[:5])}{'...' if len(dataset.numeric_columns) > 5 else ''}",
        f"- **Date Columns**: {', '.join(dataset.date_columns) if dataset.date_columns else 'None'}",
        f"- **Categorical Columns**: {', '.join(dataset.categorical_columns[:5])}{'...' if len(dataset.categorical_columns) > 5 else ''}",
        "",
    ]
    
    # Add insights if available
    if request.include_insights:
        cached_insights = session.cached_results.get(f"profile_{dataset.name}")
        if cached_insights:
            lines.extend([
                "## Data Quality",
                "",
                f"- **Missing Values**: {cached_insights.total_missing_pct:.1f}%",
                f"- **Duplicate Rows**: {cached_insights.duplicate_row_count:,} ({cached_insights.duplicate_row_pct:.1f}%)",
                "",
            ])
            
            if cached_insights.warnings:
                lines.append("### Warnings")
                lines.append("")
                for warning in cached_insights.warnings:
                    lines.append(f"- {warning}")
                lines.append("")
    
    # Add charts section (list chart titles; images are in bundle charts/ folder)
    if request.include_charts and session.charts:
        lines.extend([
            "## Charts",
            "",
        ])
        for chart_id, chart_info in session.charts.items():
            safe_title = chart_info.title.replace("|", "-")
            lines.append(f"- **{safe_title}** (type: {chart_info.chart_type})")
        lines.append("")
        lines.append("*Chart images are included in the `charts/` folder when you download the full bundle.*")
        lines.append("")
    
    # Add conversation highlights
    if session.messages:
        lines.extend([
            "## Analysis Performed",
            "",
        ])
        
        for msg in session.messages[-20:]:  # Last 20 messages
            if msg.role == "user":
                lines.append(f"**Q**: {msg.content}")
            elif msg.role == "assistant" and not msg.metadata.get("error"):
                # Truncate long responses
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                lines.append(f"**A**: {content}")
            lines.append("")
    
    # Add pinned definitions
    if session.pinned_definitions:
        lines.extend([
            "## Defined Metrics",
            "",
        ])
        for name, defn in session.pinned_definitions.items():
            lines.append(f"- **{name}**: `{defn.formula}`")
            if defn.description:
                lines.append(f"  - {defn.description}")
        lines.append("")
    
    lines.extend([
        "---",
        f"*Report generated by DataChat Analytics Platform*",
    ])
    
    return "\n".join(lines)


def _generate_html_report(session, dataset, request) -> str:
    """Generate HTML report."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{request.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 40px; }}
        .meta {{ color: #666; font-style: italic; margin-bottom: 30px; }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #4361ee; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin: 10px 0; }}
        .qa {{ margin: 20px 0; }}
        .qa-q {{ font-weight: bold; color: #16213e; }}
        .qa-a {{ margin-left: 20px; color: #555; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #4361ee; color: white; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        code {{ background: #e9ecef; padding: 2px 6px; border-radius: 4px; }}
        footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{request.title}</h1>
    <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <h2>Executive Summary</h2>
    <p>This report analyzes the <strong>{dataset.name}</strong> dataset.</p>
    
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{dataset.row_count:,}</div>
            <div class="stat-label">Rows</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{dataset.column_count}</div>
            <div class="stat-label">Columns</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(dataset.numeric_columns)}</div>
            <div class="stat-label">Numeric Columns</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(dataset.date_columns)}</div>
            <div class="stat-label">Date Columns</div>
        </div>
    </div>
"""
    
    # Add warnings
    cached_insights = session.cached_results.get(f"profile_{dataset.name}")
    if cached_insights and cached_insights.warnings:
        html += "<h2>Data Quality Warnings</h2>\n"
        for warning in cached_insights.warnings:
            html += f'<div class="warning">{warning}</div>\n'
    
    # Add charts as embedded images (so report is self-contained)
    if request.include_charts and session.charts:
        html += "<h2>Charts</h2>\n"
        for chart_id, chart_info in session.charts.items():
            b64 = visualization_service.chart_to_base64_png(chart_info)
            if b64:
                html += f'<div class="chart-block"><h3>{chart_info.title}</h3>'
                html += f'<img src="data:image/png;base64,{b64}" alt="{chart_info.title}" class="report-chart" /></div>\n'
            else:
                html += f'<div class="chart-block"><h3>{chart_info.title}</h3><p><em>Chart (export failed)</em></p></div>\n'
        html += '<style>.chart-block{margin:24px 0;}.report-chart{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;}</style>\n'
    
    # Add conversation
    if session.messages:
        html += "<h2>Analysis Performed</h2>\n"
        for msg in session.messages[-20:]:
            if msg.role == "user":
                html += f'<div class="qa"><div class="qa-q">Q: {msg.content}</div>'
            elif msg.role == "assistant" and not msg.metadata.get("error"):
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                html += f'<div class="qa-a">{content}</div></div>\n'
    
    # Add pinned definitions
    if session.pinned_definitions:
        html += "<h2>Defined Metrics</h2>\n<table><tr><th>Name</th><th>Formula</th><th>Description</th></tr>\n"
        for name, defn in session.pinned_definitions.items():
            html += f"<tr><td>{name}</td><td><code>{defn.formula}</code></td><td>{defn.description or '-'}</td></tr>\n"
        html += "</table>\n"
    
    html += """
    <footer>
        <p>Report generated by DataChat Analytics Platform</p>
    </footer>
</body>
</html>"""
    
    return html


@router.get("/bundle")
async def download_bundle(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Download everything as a ZIP bundle: dataset, charts (with images), reports (with embedded chart images),
    YData profiling (if generated), and data quality insights (if run).
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if not session.active_dataset:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    dataset = session.active_dataset
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Dataset CSV
        csv_buffer = io.StringIO()
        dataset.df.to_csv(csv_buffer, index=False)
        zf.writestr(f"data/{dataset.name}.csv", csv_buffer.getvalue())
        
        # 2. Charts: JSON + PNG for each chart (manual and chat-generated)
        for chart_id, chart_info in session.charts.items():
            safe_title = chart_info.title.replace(" ", "_").replace("/", "-")[:50]
            zf.writestr(
                f"charts/{safe_title}_{chart_id[:8]}.json",
                json.dumps(chart_info.plotly_json, indent=2)
            )
            try:
                png_bytes = await visualization_service.export_chart_png(chart_info)
                zf.writestr(f"charts/{safe_title}_{chart_id[:8]}.png", png_bytes)
            except Exception:
                pass
        
        # 3. Reports (HTML includes embedded chart images; MD lists charts)
        report_request = ReportRequest()
        html_report = _generate_html_report(session, dataset, report_request)
        zf.writestr("report.html", html_report)
        md_report = _generate_markdown_report(session, dataset, report_request)
        zf.writestr("report.md", md_report)
        
        # 4. YData profiling HTML (if generated)
        ydata_html = session.cached_results.get(f"ydata_profile_{dataset.name}")
        if ydata_html:
            zf.writestr(f"ydata_profile_{dataset.name.replace(' ', '_')}.html", ydata_html)
        
        # 5. Data quality insights JSON (if run)
        insights = session.cached_results.get(f"profile_{dataset.name}")
        if insights is not None:
            from dataclasses import asdict
            import datetime as dt
            try:
                data = asdict(insights) if hasattr(insights, "__dataclass_fields__") else dict(insights)
                def _ser(o):
                    if o is None or isinstance(o, (str, int, bool, float)):
                        return o
                    if isinstance(o, dt.datetime):
                        return o.isoformat()
                    if isinstance(o, dict):
                        return {k: _ser(v) for k, v in o.items()}
                    if isinstance(o, (list, tuple)):
                        return [_ser(x) for x in o]
                    try:
                        return float(o)
                    except (TypeError, ValueError):
                        return str(o)
                data = _ser(data)
                zf.writestr("insights_data_quality.json", json.dumps(data, indent=2))
            except Exception:
                pass
        
        # 6. README describing bundle contents
        readme_lines = [
            "DataChat Analytics – Full Export Bundle",
            "==========================================",
            "",
            "Contents:",
            "  data/         – Dataset as CSV",
            "  charts/       – All charts (manual + chat) as JSON and PNG",
            "  report.html   – Full report with embedded chart images",
            "  report.md     – Markdown report",
        ]
        if ydata_html:
            readme_lines.append(f"  ydata_profile_*.html – YData profiling analysis")
        if insights is not None:
            readme_lines.append("  insights_data_quality.json – Data quality / health insights")
        readme_lines.extend(["", "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M")])
        zf.writestr("README.txt", "\n".join(readme_lines))
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=datachat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        }
    )


@router.get("/charts")
async def list_charts(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    List all charts in the session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {
        "charts": [
            {
                "id": chart_id,
                "type": chart_info.chart_type,
                "title": chart_info.title,
                "created_at": chart_info.created_at.isoformat()
            }
            for chart_id, chart_info in session.charts.items()
        ]
    }
