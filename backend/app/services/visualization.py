"""
Visualization Service
Generates charts using Plotly with export capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from io import BytesIO
import base64

from app.core.session_manager import ChartInfo
from datetime import datetime
from uuid import uuid4


class VisualizationService:
    """
    Service for generating interactive and exportable visualizations.
    """
    
    # Theme presets
    THEMES = {
        "clean": {
            "template": "plotly_white",
            "font_family": "Inter, sans-serif",
            "title_font_size": 18,
            "axis_font_size": 12,
            "colors": px.colors.qualitative.Set2
        },
        "dark": {
            "template": "plotly_dark",
            "font_family": "Inter, sans-serif",
            "title_font_size": 18,
            "axis_font_size": 12,
            "colors": px.colors.qualitative.Dark24
        },
        "presentation": {
            "template": "plotly_white",
            "font_family": "Arial, sans-serif",
            "title_font_size": 24,
            "axis_font_size": 14,
            "colors": px.colors.qualitative.Bold
        }
    }
    
    def __init__(self):
        pass
    
    def _apply_theme(self, fig: go.Figure, theme: str = "clean") -> go.Figure:
        """Apply a theme preset to a figure."""
        theme_config = self.THEMES.get(theme, self.THEMES["clean"])
        
        fig.update_layout(
            template=theme_config["template"],
            font_family=theme_config["font_family"],
            title_font_size=theme_config["title_font_size"],
            xaxis_title_font_size=theme_config["axis_font_size"],
            yaxis_title_font_size=theme_config["axis_font_size"],
        )
        
        return fig
    
    def _create_chart_info(
        self,
        fig: go.Figure,
        chart_type: str,
        title: str,
        underlying_data: Optional[pd.DataFrame] = None
    ) -> ChartInfo:
        """Create a ChartInfo object from a figure."""
        return ChartInfo(
            id=str(uuid4()),
            chart_type=chart_type,
            title=title,
            created_at=datetime.now(),
            plotly_json=json.loads(fig.to_json()),
            underlying_data=underlying_data
        )
    
    # ==================== Core Chart Types ====================
    
    async def create_line_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a line chart for time trends."""
        fig = px.line(
            df, x=x, y=y, color=color,
            title=title or f"{y} over {x}",
            markers=True
        )
        
        fig = self._apply_theme(fig, theme)
        fig.update_traces(line=dict(width=2))
        
        return self._create_chart_info(fig, "line", title or f"{y} over {x}", df[[x, y] + ([color] if color else [])])
    
    async def create_bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        orientation: str = 'v',  # 'v' or 'h'
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a bar chart."""
        if orientation == 'h':
            fig = px.bar(df, y=x, x=y, color=color, orientation='h',
                        title=title or f"{y} by {x}")
        else:
            fig = px.bar(df, x=x, y=y, color=color,
                        title=title or f"{y} by {x}")
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "bar", title or f"{y} by {x}", df)
    
    async def create_stacked_bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: str,
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a stacked bar chart."""
        fig = px.bar(
            df, x=x, y=y, color=color,
            title=title or f"{y} by {x} (stacked by {color})",
            barmode='stack'
        )
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "stacked_bar", title or f"Stacked {y} by {x}", df)
    
    async def create_histogram(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int = 30,
        color: Optional[str] = None,
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a histogram for distributions."""
        fig = px.histogram(
            df, x=column, nbins=bins, color=color,
            title=title or f"Distribution of {column}",
            marginal="box"  # Add box plot above
        )
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "histogram", title or f"Distribution of {column}", df[[column]])
    
    async def create_box_plot(
        self,
        df: pd.DataFrame,
        column: str,
        group_by: Optional[str] = None,
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a box plot for outlier visualization."""
        if group_by:
            fig = px.box(df, x=group_by, y=column,
                        title=title or f"{column} by {group_by}")
        else:
            fig = px.box(df, y=column,
                        title=title or f"Box Plot of {column}")
        
        fig = self._apply_theme(fig, theme)
        fig.update_traces(boxpoints='outliers')
        
        return self._create_chart_info(fig, "box", title or f"Box Plot of {column}", df)
    
    async def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        trendline: Optional[str] = None,  # 'ols', 'lowess'
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a scatter plot for relationships."""
        fig = px.scatter(
            df, x=x, y=y, color=color, size=size,
            trendline=trendline,
            title=title or f"{y} vs {x}"
        )
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "scatter", title or f"{y} vs {x}", df)
    
    async def create_heatmap(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        color_scale: str = "RdBu_r",
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a heatmap (for correlations, pivots, etc.)."""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=list(data.columns),
            y=list(data.index),
            colorscale=color_scale,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title or "Heatmap",
            xaxis_title="",
            yaxis_title=""
        )
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "heatmap", title or "Heatmap", data)
    
    async def create_area_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create an area chart (cumulative)."""
        fig = px.area(
            df, x=x, y=y, color=color,
            title=title or f"Cumulative {y}"
        )
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "area", title or f"Cumulative {y}", df)
    
    async def create_pie_chart(
        self,
        df: pd.DataFrame,
        names: str,
        values: str,
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a pie chart."""
        fig = px.pie(
            df, names=names, values=values,
            title=title or f"{values} by {names}"
        )
        
        fig = self._apply_theme(fig, theme)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return self._create_chart_info(fig, "pie", title or f"{values} by {names}", df)
    
    # ==================== Advanced Chart Types ====================
    
    async def create_pareto_chart(
        self,
        df: pd.DataFrame,
        category: str,
        value: str,
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a Pareto chart (80/20)."""
        # Sort by value descending
        sorted_df = df.groupby(category)[value].sum().sort_values(ascending=False).reset_index()
        sorted_df['cumulative'] = sorted_df[value].cumsum()
        sorted_df['cumulative_pct'] = sorted_df['cumulative'] / sorted_df[value].sum() * 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=sorted_df[category], y=sorted_df[value], name=value),
            secondary_y=False
        )
        
        # Line for cumulative percentage
        fig.add_trace(
            go.Scatter(x=sorted_df[category], y=sorted_df['cumulative_pct'],
                      mode='lines+markers', name='Cumulative %'),
            secondary_y=True
        )
        
        # Add 80% line
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                     annotation_text="80%", secondary_y=True)
        
        fig.update_layout(title=title or f"Pareto Chart: {value} by {category}")
        fig.update_yaxes(title_text=value, secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "pareto", title or "Pareto Chart", sorted_df)
    
    async def create_funnel_chart(
        self,
        stages: List[str],
        values: List[int],
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a funnel chart."""
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textposition="inside",
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(title=title or "Funnel Analysis")
        fig = self._apply_theme(fig, theme)
        
        data = pd.DataFrame({"stage": stages, "count": values})
        return self._create_chart_info(fig, "funnel", title or "Funnel Analysis", data)
    
    async def create_cohort_heatmap(
        self,
        cohort_data: List[Dict[str, Any]],
        title: Optional[str] = None,
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a cohort retention heatmap."""
        # Convert cohort data to matrix
        cohorts = [d['cohort'] for d in cohort_data]
        max_periods = max(max(d['retention'].keys()) for d in cohort_data if d['retention'])
        
        z = []
        for cohort in cohort_data:
            row = [cohort['retention'].get(i, None) for i in range(max_periods + 1)]
            z.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[f"Period {i}" for i in range(max_periods + 1)],
            y=cohorts,
            colorscale="Blues",
            hoverongaps=False,
            text=[[f"{v:.1f}%" if v else "" for v in row] for row in z],
            texttemplate="%{text}",
            showscale=True
        ))
        
        fig.update_layout(
            title=title or "Cohort Retention",
            xaxis_title="Periods Since Cohort",
            yaxis_title="Cohort"
        )
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "cohort_heatmap", title or "Cohort Retention", None)
    
    async def create_kpi_cards(
        self,
        cards: List[Dict[str, Any]],
        theme: str = "clean"
    ) -> ChartInfo:
        """Create a KPI card visualization."""
        n_cards = len(cards)
        cols = min(4, n_cards)
        rows = (n_cards + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{"type": "indicator"} for _ in range(cols)] for _ in range(rows)],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        for i, card in enumerate(cards):
            row = i // cols + 1
            col = i % cols + 1
            
            # Determine mode
            mode = "number"
            if card.get("change_pct") is not None:
                mode = "number+delta"
            
            indicator = go.Indicator(
                mode=mode,
                value=card["value"],
                title={"text": card["name"]},
                number={"valueformat": ",.2f" if isinstance(card["value"], float) else ","},
            )
            
            if card.get("change_pct") is not None:
                indicator.delta = {
                    "reference": card["value"] / (1 + card["change_pct"] / 100),
                    "relative": True,
                    "valueformat": ".1%"
                }
            
            fig.add_trace(indicator, row=row, col=col)
        
        fig.update_layout(
            title="Key Performance Indicators",
            height=150 * rows
        )
        
        fig = self._apply_theme(fig, theme)
        
        return self._create_chart_info(fig, "kpi_cards", "KPI Cards", None)
    
    # ==================== Auto Chart Selection ====================
    
    async def auto_chart(
        self,
        df: pd.DataFrame,
        x: Optional[str] = None,
        y: Optional[str] = None,
        query_type: str = "general",  # 'trend', 'distribution', 'comparison', 'relationship'
        theme: str = "clean"
    ) -> ChartInfo:
        """
        Automatically select the best chart type based on data and query.
        """
        # Infer column types
        x_type = self._infer_column_type(df[x]) if x else None
        y_type = self._infer_column_type(df[y]) if y else None
        
        # Decision logic
        if query_type == 'trend' and x_type == 'datetime':
            return await self.create_line_chart(df, x, y, theme=theme)
        
        if query_type == 'distribution':
            return await self.create_histogram(df, y or x, theme=theme)
        
        if query_type == 'comparison' and x_type == 'categorical':
            return await self.create_bar_chart(df, x, y, theme=theme)
        
        if query_type == 'relationship' and x_type == 'numeric' and y_type == 'numeric':
            return await self.create_scatter_plot(df, x, y, trendline='ols', theme=theme)
        
        # Default logic based on column types
        if x_type == 'datetime':
            return await self.create_line_chart(df, x, y, theme=theme)
        
        if x_type == 'categorical' and y_type == 'numeric':
            return await self.create_bar_chart(df, x, y, theme=theme)
        
        if x_type == 'numeric' and y_type == 'numeric':
            return await self.create_scatter_plot(df, x, y, theme=theme)
        
        # Fallback: bar chart
        return await self.create_bar_chart(df, x, y, theme=theme)
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer high-level column type for chart selection."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        if series.nunique() < 20:
            return 'categorical'
        return 'text'
    
    # ==================== Export Functions ====================
    
    async def export_chart_png(
        self,
        chart_info: ChartInfo,
        width: int = 1200,
        height: int = 800
    ) -> bytes:
        """Export chart as PNG bytes."""
        fig = go.Figure(chart_info.plotly_json)
        
        try:
            img_bytes = fig.to_image(format="png", width=width, height=height)
            return img_bytes
        except Exception as e:
            # Fallback if kaleido not available
            raise ValueError(f"PNG export failed: {str(e)}. Ensure kaleido is installed.")
    
    async def export_chart_svg(
        self,
        chart_info: ChartInfo,
        width: int = 1200,
        height: int = 800
    ) -> str:
        """Export chart as SVG string."""
        fig = go.Figure(chart_info.plotly_json)
        
        try:
            svg_str = fig.to_image(format="svg", width=width, height=height)
            return svg_str.decode('utf-8')
        except Exception as e:
            raise ValueError(f"SVG export failed: {str(e)}")
    
    async def export_chart_data(
        self,
        chart_info: ChartInfo
    ) -> Optional[bytes]:
        """Export underlying chart data as CSV."""
        if chart_info.underlying_data is not None:
            return chart_info.underlying_data.to_csv(index=False).encode('utf-8')
        return None
    
    def chart_to_base64_png(self, chart_info: ChartInfo) -> str:
        """Convert chart to base64-encoded PNG for embedding."""
        fig = go.Figure(chart_info.plotly_json)
        try:
            img_bytes = fig.to_image(format="png", width=800, height=500)
            return base64.b64encode(img_bytes).decode('utf-8')
        except:
            return None


# Singleton instance
visualization_service = VisualizationService()
