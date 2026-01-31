"""
Analytics API Routes
Advanced analytics: KPIs, trends, cohorts, funnels, anomaly detection.
"""

from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel
from typing import Optional, List

from app.core.session_manager import session_manager
from app.services.analytics_engine import analytics_engine
from app.services.visualization import visualization_service


router = APIRouter()


class KPIRequest(BaseModel):
    """Request for KPI analysis."""
    metric_column: str
    date_column: Optional[str] = None
    group_column: Optional[str] = None


class ContributionRequest(BaseModel):
    """Request for contribution analysis."""
    metric_column: str
    segment_column: str
    top_n: int = 10


class TimeSeriesRequest(BaseModel):
    """Request for time series analysis."""
    metric_column: str
    date_column: str
    period: str = 'M'  # D, W, M, Q, Y


class CohortRequest(BaseModel):
    """Request for cohort analysis."""
    user_column: str
    date_column: str
    metric_column: Optional[str] = None
    period: str = 'M'


class FunnelRequest(BaseModel):
    """Request for funnel analysis."""
    stage_column: str
    stages: List[str]
    count_column: Optional[str] = None
    timestamp_column: Optional[str] = None


class AnomalyRequest(BaseModel):
    """Request for anomaly detection."""
    metric_column: str
    date_column: Optional[str] = None
    method: str = 'iqr'  # 'iqr', 'zscore'
    sensitivity: float = 1.5


class DriverRequest(BaseModel):
    """Request for driver analysis."""
    target_column: str
    feature_columns: List[str]
    method: str = 'correlation'  # 'correlation', 'importance'


@router.post("/kpi")
async def compute_kpis(
    request: KPIRequest,
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Compute KPI cards for a metric.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    dataset = session.datasets.get(dataset_name or session.active_dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if request.metric_column not in dataset.df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{request.metric_column}' not found")
    
    try:
        result = await analytics_engine.compute_kpi_cards(
            dataset.df,
            request.metric_column,
            request.date_column,
            request.group_column
        )
        
        # Generate KPI card visualization
        if result.get("cards"):
            chart = await visualization_service.create_kpi_cards(result["cards"])
            session.charts[chart.id] = chart
            result["chart"] = {
                "id": chart.id,
                "plotly_json": chart.plotly_json
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KPI analysis failed: {str(e)}")


@router.post("/contribution")
async def analyze_contribution(
    request: ContributionRequest,
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Analyze segment contributions to a metric.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    dataset = session.datasets.get(dataset_name or session.active_dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        result = await analytics_engine.contribution_analysis(
            dataset.df,
            request.metric_column,
            request.segment_column,
            request.top_n
        )
        
        # Generate Pareto chart
        chart = await visualization_service.create_pareto_chart(
            dataset.df,
            request.segment_column,
            request.metric_column,
            title=f"Contribution Analysis: {request.metric_column} by {request.segment_column}"
        )
        session.charts[chart.id] = chart
        result["chart"] = {
            "id": chart.id,
            "plotly_json": chart.plotly_json
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contribution analysis failed: {str(e)}")


@router.post("/time-series")
async def analyze_time_series(
    request: TimeSeriesRequest,
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Perform time series analysis with trend detection.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    dataset = session.datasets.get(dataset_name or session.active_dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        result = await analytics_engine.time_series_analysis(
            dataset.df,
            request.metric_column,
            request.date_column,
            request.period
        )
        
        # Generate time series chart
        import pandas as pd
        ts_data = pd.DataFrame(result["time_series"])
        ts_data.columns = ['period', request.metric_column]
        
        chart = await visualization_service.create_line_chart(
            ts_data,
            'period',
            request.metric_column,
            title=f"{request.metric_column} Over Time"
        )
        session.charts[chart.id] = chart
        result["chart"] = {
            "id": chart.id,
            "plotly_json": chart.plotly_json
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time series analysis failed: {str(e)}")


@router.post("/cohort")
async def analyze_cohorts(
    request: CohortRequest,
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Perform cohort retention analysis.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    dataset = session.datasets.get(dataset_name or session.active_dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        result = await analytics_engine.cohort_analysis(
            dataset.df,
            request.user_column,
            request.date_column,
            request.metric_column,
            request.period
        )
        
        # Generate cohort heatmap
        if result.get("cohorts"):
            chart = await visualization_service.create_cohort_heatmap(
                result["cohorts"],
                title="Cohort Retention Analysis"
            )
            session.charts[chart.id] = chart
            result["chart"] = {
                "id": chart.id,
                "plotly_json": chart.plotly_json
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohort analysis failed: {str(e)}")


@router.post("/funnel")
async def analyze_funnel(
    request: FunnelRequest,
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Perform funnel analysis.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    dataset = session.datasets.get(dataset_name or session.active_dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        result = await analytics_engine.funnel_analysis(
            dataset.df,
            request.stage_column,
            request.stages,
            request.count_column,
            request.timestamp_column
        )
        
        # Generate funnel chart
        stages = [s["stage"] for s in result["stages"]]
        values = [s["count"] for s in result["stages"]]
        
        chart = await visualization_service.create_funnel_chart(
            stages,
            values,
            title="Funnel Analysis"
        )
        session.charts[chart.id] = chart
        result["chart"] = {
            "id": chart.id,
            "plotly_json": chart.plotly_json
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funnel analysis failed: {str(e)}")


@router.post("/anomalies")
async def detect_anomalies(
    request: AnomalyRequest,
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Detect anomalies in data.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    dataset = session.datasets.get(dataset_name or session.active_dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        result = await analytics_engine.detect_anomalies(
            dataset.df,
            request.metric_column,
            request.date_column,
            request.method,
            request.sensitivity
        )
        
        # Generate box plot to show outliers
        chart = await visualization_service.create_box_plot(
            dataset.df,
            request.metric_column,
            title=f"Outlier Detection: {request.metric_column}"
        )
        session.charts[chart.id] = chart
        result["chart"] = {
            "id": chart.id,
            "plotly_json": chart.plotly_json
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@router.post("/drivers")
async def analyze_drivers(
    request: DriverRequest,
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None)
):
    """
    Identify key drivers of a target metric.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    dataset = session.datasets.get(dataset_name or session.active_dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        result = await analytics_engine.driver_analysis(
            dataset.df,
            request.target_column,
            request.feature_columns,
            request.method
        )
        
        # Generate bar chart of drivers
        if result.get("drivers"):
            import pandas as pd
            drivers_df = pd.DataFrame(result["drivers"])
            
            value_col = 'correlation' if request.method == 'correlation' else 'importance'
            drivers_df['abs_value'] = drivers_df[value_col].abs()
            drivers_df = drivers_df.sort_values('abs_value', ascending=True)
            
            chart = await visualization_service.create_bar_chart(
                drivers_df,
                x=value_col,
                y='feature',
                orientation='h',
                title=f"Drivers of {request.target_column}"
            )
            session.charts[chart.id] = chart
            result["chart"] = {
                "id": chart.id,
                "plotly_json": chart.plotly_json
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Driver analysis failed: {str(e)}")


@router.get("/correlation-matrix")
async def get_correlation_matrix(
    session_id: str = Header(..., alias="X-Session-ID"),
    dataset_name: Optional[str] = Query(None),
    columns: Optional[str] = Query(None, description="Comma-separated column names")
):
    """
    Get correlation matrix for numeric columns.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    dataset = session.datasets.get(dataset_name or session.active_dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        import pandas as pd
        
        if columns:
            cols = [c.strip() for c in columns.split(',')]
        else:
            cols = dataset.numeric_columns[:10]  # Limit to 10 columns
        
        numeric_df = dataset.df[cols].apply(pd.to_numeric, errors='coerce')
        corr_matrix = numeric_df.corr()
        
        # Generate heatmap
        chart = await visualization_service.create_heatmap(
            corr_matrix,
            title="Correlation Matrix"
        )
        session.charts[chart.id] = chart
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "columns": cols,
            "chart": {
                "id": chart.id,
                "plotly_json": chart.plotly_json
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")
