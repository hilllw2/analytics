"""
Analytics Engine
Advanced analytics modules: KPI analysis, trends, cohorts, funnels, anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.session_manager import DatasetInfo, Session


@dataclass
class KPICard:
    """A KPI metric card."""
    name: str
    value: float
    formatted_value: str
    change_pct: Optional[float] = None
    change_period: Optional[str] = None
    trend: Optional[str] = None  # 'up', 'down', 'stable'


class AnalyticsEngine:
    """
    Advanced analytics capabilities.
    """
    
    def __init__(self):
        pass
    
    # ==================== KPI & Performance Analysis ====================
    
    async def compute_kpi_cards(
        self,
        df: pd.DataFrame,
        metric_column: str,
        date_column: Optional[str] = None,
        group_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute KPI cards for a metric.
        
        Returns cards for: Total, Average, Min, Max, and time-based changes if date exists.
        """
        numeric = pd.to_numeric(df[metric_column], errors='coerce').dropna()
        
        cards = []
        
        # Total
        total = numeric.sum()
        cards.append(KPICard(
            name=f"Total {metric_column}",
            value=total,
            formatted_value=self._format_number(total)
        ))
        
        # Average
        avg = numeric.mean()
        cards.append(KPICard(
            name=f"Average {metric_column}",
            value=avg,
            formatted_value=self._format_number(avg)
        ))
        
        # Median
        median = numeric.median()
        cards.append(KPICard(
            name=f"Median {metric_column}",
            value=median,
            formatted_value=self._format_number(median)
        ))
        
        # Count
        count = len(numeric)
        cards.append(KPICard(
            name="Record Count",
            value=count,
            formatted_value=f"{count:,}"
        ))
        
        # Time-based metrics
        if date_column:
            time_cards = await self._compute_time_kpis(df, metric_column, date_column)
            cards.extend(time_cards)
        
        return {
            "cards": [self._kpi_to_dict(c) for c in cards],
            "metric": metric_column,
            "date_column": date_column
        }
    
    async def _compute_time_kpis(
        self,
        df: pd.DataFrame,
        metric_column: str,
        date_column: str
    ) -> List[KPICard]:
        """Compute time-based KPI cards (MoM, YoY, etc.)."""
        cards = []
        
        try:
            df_time = df.copy()
            df_time['_date'] = pd.to_datetime(df_time[date_column], errors='coerce')
            df_time = df_time.dropna(subset=['_date', metric_column])
            
            if len(df_time) < 2:
                return cards
            
            # Get date range
            max_date = df_time['_date'].max()
            
            # Monthly change
            current_month = df_time[df_time['_date'] >= max_date - timedelta(days=30)]
            prev_month = df_time[
                (df_time['_date'] >= max_date - timedelta(days=60)) &
                (df_time['_date'] < max_date - timedelta(days=30))
            ]
            
            if len(current_month) > 0 and len(prev_month) > 0:
                current_sum = current_month[metric_column].sum()
                prev_sum = prev_month[metric_column].sum()
                
                if prev_sum > 0:
                    mom_change = (current_sum / prev_sum - 1) * 100
                    cards.append(KPICard(
                        name="Month-over-Month",
                        value=mom_change,
                        formatted_value=f"{mom_change:+.1f}%",
                        change_pct=mom_change,
                        change_period="vs last 30 days",
                        trend="up" if mom_change > 0 else "down"
                    ))
            
            # Year-over-Year (if data spans > 1 year)
            if (max_date - df_time['_date'].min()).days > 365:
                current_year = df_time[df_time['_date'] >= max_date - timedelta(days=365)]
                prev_year = df_time[
                    (df_time['_date'] >= max_date - timedelta(days=730)) &
                    (df_time['_date'] < max_date - timedelta(days=365))
                ]
                
                if len(current_year) > 0 and len(prev_year) > 0:
                    current_sum = current_year[metric_column].sum()
                    prev_sum = prev_year[metric_column].sum()
                    
                    if prev_sum > 0:
                        yoy_change = (current_sum / prev_sum - 1) * 100
                        cards.append(KPICard(
                            name="Year-over-Year",
                            value=yoy_change,
                            formatted_value=f"{yoy_change:+.1f}%",
                            change_pct=yoy_change,
                            change_period="vs same period last year",
                            trend="up" if yoy_change > 0 else "down"
                        ))
        
        except Exception:
            pass
        
        return cards
    
    def _kpi_to_dict(self, kpi: KPICard) -> Dict[str, Any]:
        """Convert KPICard to dictionary."""
        return {
            "name": kpi.name,
            "value": kpi.value,
            "formatted_value": kpi.formatted_value,
            "change_pct": kpi.change_pct,
            "change_period": kpi.change_period,
            "trend": kpi.trend
        }
    
    def _format_number(self, value: float) -> str:
        """Format a number for display."""
        if abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B"
        elif abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.2f}K"
        elif isinstance(value, float):
            return f"{value:,.2f}"
        else:
            return f"{value:,}"
    
    # ==================== Contribution Analysis ====================
    
    async def contribution_analysis(
        self,
        df: pd.DataFrame,
        metric_column: str,
        segment_column: str,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze which segments contribute most to a metric.
        """
        agg = df.groupby(segment_column)[metric_column].agg(['sum', 'count', 'mean'])
        agg = agg.sort_values('sum', ascending=False)
        
        total = agg['sum'].sum()
        agg['contribution_pct'] = agg['sum'] / total * 100
        agg['cumulative_pct'] = agg['contribution_pct'].cumsum()
        
        top_segments = agg.head(top_n).reset_index()
        
        # Find 80/20 point
        pareto_point = agg[agg['cumulative_pct'] <= 80]
        segments_for_80 = len(pareto_point) + 1
        
        return {
            "segments": top_segments.to_dict(orient='records'),
            "total": total,
            "top_segment": {
                "name": str(agg.index[0]),
                "value": float(agg.iloc[0]['sum']),
                "contribution_pct": float(agg.iloc[0]['contribution_pct'])
            },
            "pareto": {
                "segments_for_80_pct": segments_for_80,
                "total_segments": len(agg)
            }
        }
    
    # ==================== Trend & Time Series Analysis ====================
    
    async def time_series_analysis(
        self,
        df: pd.DataFrame,
        metric_column: str,
        date_column: str,
        period: str = 'M'  # D, W, M, Q, Y
    ) -> Dict[str, Any]:
        """
        Perform time series analysis with trend detection.
        """
        df_time = df.copy()
        df_time['_date'] = pd.to_datetime(df_time[date_column], errors='coerce')
        df_time = df_time.dropna(subset=['_date', metric_column])
        
        # Aggregate by period
        df_time['_period'] = df_time['_date'].dt.to_period(period)
        time_series = df_time.groupby('_period')[metric_column].sum().reset_index()
        time_series['_period_str'] = time_series['_period'].astype(str)
        
        values = time_series[metric_column].values
        
        result = {
            "time_series": time_series[['_period_str', metric_column]].to_dict(orient='records'),
            "period": period,
            "data_points": len(time_series)
        }
        
        if len(values) >= 3:
            # Moving average
            window = min(3, len(values))
            ma = pd.Series(values).rolling(window=window).mean().tolist()
            result["moving_average"] = ma
            result["ma_window"] = window
            
            # Linear trend
            X = np.arange(len(values)).reshape(-1, 1)
            y = values
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            result["trend"] = {
                "slope": float(slope),
                "direction": "increasing" if slope > 0 else "decreasing",
                "strength": "strong" if abs(slope) > np.std(values) else "weak"
            }
            
            # Change point detection (simple)
            if len(values) >= 6:
                mid = len(values) // 2
                first_half_mean = np.mean(values[:mid])
                second_half_mean = np.mean(values[mid:])
                
                change = (second_half_mean - first_half_mean) / first_half_mean * 100 if first_half_mean > 0 else 0
                result["change_point"] = {
                    "first_half_avg": float(first_half_mean),
                    "second_half_avg": float(second_half_mean),
                    "change_pct": float(change)
                }
            
            # Seasonality hint (if enough data)
            if len(values) >= 12:
                # Simple seasonality check using autocorrelation
                autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                # Look for peaks (potential seasonal periods)
                potential_periods = []
                for i in range(2, min(13, len(autocorr))):
                    if autocorr[i] > 0.3:
                        potential_periods.append({"period": i, "strength": float(autocorr[i])})
                
                if potential_periods:
                    result["seasonality_hints"] = potential_periods[:3]
        
        return result
    
    # ==================== Cohort Analysis ====================
    
    async def cohort_analysis(
        self,
        df: pd.DataFrame,
        user_column: str,
        date_column: str,
        metric_column: Optional[str] = None,
        period: str = 'M'
    ) -> Dict[str, Any]:
        """
        Perform cohort retention analysis.
        """
        df_cohort = df.copy()
        df_cohort['_date'] = pd.to_datetime(df_cohort[date_column], errors='coerce')
        df_cohort = df_cohort.dropna(subset=['_date', user_column])
        
        # Determine cohort (first activity period)
        df_cohort['_period'] = df_cohort['_date'].dt.to_period(period)
        first_activity = df_cohort.groupby(user_column)['_period'].min().reset_index()
        first_activity.columns = [user_column, '_cohort']
        
        df_cohort = df_cohort.merge(first_activity, on=user_column)
        
        # Calculate period index (periods since cohort)
        df_cohort['_period_index'] = (df_cohort['_period'] - df_cohort['_cohort']).apply(lambda x: x.n)
        
        # Build cohort matrix
        cohort_data = df_cohort.groupby(['_cohort', '_period_index'])[user_column].nunique().reset_index()
        cohort_data.columns = ['cohort', 'period_index', 'users']
        
        # Pivot to matrix
        cohort_matrix = cohort_data.pivot(index='cohort', columns='period_index', values='users')
        
        # Calculate retention rates
        retention_matrix = cohort_matrix.div(cohort_matrix[0], axis=0) * 100
        
        # Convert to serializable format
        cohort_sizes = cohort_matrix[0].to_dict()
        retention_data = []
        
        for cohort in retention_matrix.index:
            row = {
                "cohort": str(cohort),
                "cohort_size": int(cohort_sizes.get(cohort, 0)),
                "retention": {}
            }
            for period_idx in retention_matrix.columns:
                val = retention_matrix.loc[cohort, period_idx]
                if not pd.isna(val):
                    row["retention"][int(period_idx)] = round(val, 1)
            retention_data.append(row)
        
        # Calculate average retention by period
        avg_retention = retention_matrix.mean().to_dict()
        avg_retention = {int(k): round(v, 1) for k, v in avg_retention.items() if not pd.isna(v)}
        
        return {
            "cohorts": retention_data,
            "average_retention": avg_retention,
            "period": period,
            "total_users": df_cohort[user_column].nunique(),
            "total_cohorts": len(retention_data)
        }
    
    # ==================== Funnel Analysis ====================
    
    async def funnel_analysis(
        self,
        df: pd.DataFrame,
        stage_column: str,
        stages: List[str],
        count_column: Optional[str] = None,
        timestamp_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform funnel analysis.
        
        Args:
            df: DataFrame with funnel data
            stage_column: Column containing stage names
            stages: Ordered list of stage names
            count_column: Optional column to sum for each stage (defaults to row count)
            timestamp_column: Optional timestamp for stage duration analysis
        """
        funnel_data = []
        prev_count = None
        
        for i, stage in enumerate(stages):
            stage_df = df[df[stage_column] == stage]
            
            if count_column:
                count = stage_df[count_column].sum()
            else:
                count = len(stage_df)
            
            stage_data = {
                "stage": stage,
                "stage_index": i,
                "count": int(count),
                "conversion_from_start": None,
                "conversion_from_prev": None,
                "drop_off_count": None,
                "drop_off_pct": None
            }
            
            if i == 0:
                stage_data["conversion_from_start"] = 100.0
                stage_data["conversion_from_prev"] = 100.0
            else:
                first_count = funnel_data[0]["count"]
                stage_data["conversion_from_start"] = round(count / first_count * 100, 1) if first_count > 0 else 0
                stage_data["conversion_from_prev"] = round(count / prev_count * 100, 1) if prev_count > 0 else 0
                
                stage_data["drop_off_count"] = prev_count - count
                stage_data["drop_off_pct"] = round((prev_count - count) / prev_count * 100, 1) if prev_count > 0 else 0
            
            prev_count = count
            funnel_data.append(stage_data)
        
        # Stage duration (if timestamp available)
        durations = None
        if timestamp_column:
            durations = await self._compute_stage_durations(df, stage_column, stages, timestamp_column)
        
        # Find biggest drop-off
        biggest_drop = max(funnel_data[1:], key=lambda x: x["drop_off_pct"] or 0)
        
        return {
            "stages": funnel_data,
            "overall_conversion": funnel_data[-1]["conversion_from_start"],
            "biggest_drop_off": {
                "from_stage": stages[biggest_drop["stage_index"] - 1],
                "to_stage": biggest_drop["stage"],
                "drop_off_pct": biggest_drop["drop_off_pct"]
            },
            "durations": durations
        }
    
    async def _compute_stage_durations(
        self,
        df: pd.DataFrame,
        stage_column: str,
        stages: List[str],
        timestamp_column: str
    ) -> List[Dict[str, Any]]:
        """Compute time between funnel stages."""
        # This would require entity-level data (e.g., user_id) to compute properly
        # Simplified version: average time in each stage
        return None
    
    # ==================== Anomaly Detection ====================
    
    async def detect_anomalies(
        self,
        df: pd.DataFrame,
        metric_column: str,
        date_column: Optional[str] = None,
        method: str = 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
        sensitivity: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detect anomalies in data.
        """
        numeric = pd.to_numeric(df[metric_column], errors='coerce')
        
        if method == 'iqr':
            anomalies = self._detect_iqr_anomalies(numeric, sensitivity)
        elif method == 'zscore':
            anomalies = self._detect_zscore_anomalies(numeric, sensitivity)
        else:
            anomalies = self._detect_iqr_anomalies(numeric, sensitivity)
        
        # Get anomaly indices
        anomaly_mask = anomalies['is_anomaly']
        anomaly_df = df[anomaly_mask].copy()
        anomaly_df['_anomaly_type'] = anomalies['anomaly_type'][anomaly_mask]
        anomaly_df['_anomaly_score'] = anomalies['score'][anomaly_mask]
        
        # Time-based anomaly detection
        time_anomalies = None
        if date_column:
            time_anomalies = await self._detect_time_anomalies(df, metric_column, date_column)
        
        return {
            "method": method,
            "total_records": len(df),
            "anomaly_count": int(anomaly_mask.sum()),
            "anomaly_pct": round(anomaly_mask.sum() / len(df) * 100, 2),
            "anomalies": anomaly_df.head(100).to_dict(orient='records'),
            "bounds": {
                "lower": float(anomalies['lower_bound']),
                "upper": float(anomalies['upper_bound'])
            },
            "time_anomalies": time_anomalies,
            "summary": {
                "high_anomalies": int((anomalies['anomaly_type'] == 'high').sum()),
                "low_anomalies": int((anomalies['anomaly_type'] == 'low').sum())
            }
        }
    
    def _detect_iqr_anomalies(
        self,
        series: pd.Series,
        multiplier: float = 1.5
    ) -> Dict[str, Any]:
        """Detect anomalies using IQR method."""
        clean = series.dropna()
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        
        is_anomaly = (series < lower) | (series > upper)
        anomaly_type = pd.Series(['normal'] * len(series), index=series.index)
        anomaly_type[series < lower] = 'low'
        anomaly_type[series > upper] = 'high'
        
        # Score based on distance from bounds
        score = pd.Series(0.0, index=series.index)
        score[series < lower] = (lower - series[series < lower]) / iqr
        score[series > upper] = (series[series > upper] - upper) / iqr
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'score': score,
            'lower_bound': lower,
            'upper_bound': upper
        }
    
    def _detect_zscore_anomalies(
        self,
        series: pd.Series,
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Detect anomalies using Z-score method."""
        clean = series.dropna()
        mean = clean.mean()
        std = clean.std()
        
        if std == 0:
            return {
                'is_anomaly': pd.Series([False] * len(series), index=series.index),
                'anomaly_type': pd.Series(['normal'] * len(series), index=series.index),
                'score': pd.Series([0.0] * len(series), index=series.index),
                'lower_bound': mean,
                'upper_bound': mean
            }
        
        zscore = (series - mean) / std
        is_anomaly = abs(zscore) > threshold
        
        anomaly_type = pd.Series(['normal'] * len(series), index=series.index)
        anomaly_type[zscore < -threshold] = 'low'
        anomaly_type[zscore > threshold] = 'high'
        
        lower = mean - threshold * std
        upper = mean + threshold * std
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'score': abs(zscore),
            'lower_bound': lower,
            'upper_bound': upper
        }
    
    async def _detect_time_anomalies(
        self,
        df: pd.DataFrame,
        metric_column: str,
        date_column: str
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in time series (spikes/drops)."""
        df_time = df.copy()
        df_time['_date'] = pd.to_datetime(df_time[date_column], errors='coerce')
        df_time = df_time.dropna(subset=['_date', metric_column])
        df_time = df_time.sort_values('_date')
        
        # Daily aggregation
        daily = df_time.groupby(df_time['_date'].dt.date)[metric_column].sum()
        
        if len(daily) < 7:
            return None
        
        # Compute rolling statistics
        rolling_mean = daily.rolling(window=7, min_periods=3).mean()
        rolling_std = daily.rolling(window=7, min_periods=3).std()
        
        # Detect spikes/drops
        anomalies = []
        for date, value in daily.items():
            rm = rolling_mean.get(date)
            rs = rolling_std.get(date)
            
            if rm is None or rs is None or rs == 0:
                continue
            
            zscore = (value - rm) / rs
            if abs(zscore) > 2:
                anomalies.append({
                    "date": str(date),
                    "value": float(value),
                    "expected": float(rm),
                    "deviation": float(zscore),
                    "type": "spike" if zscore > 0 else "drop"
                })
        
        return anomalies[:20]  # Top 20
    
    # ==================== Driver Analysis ====================
    
    async def driver_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        method: str = 'correlation'  # 'correlation', 'importance'
    ) -> Dict[str, Any]:
        """
        Identify key drivers of a target metric.
        """
        # Clean data
        df_clean = df[[target_column] + feature_columns].dropna()
        
        if len(df_clean) < 10:
            return {"error": "Insufficient data for driver analysis"}
        
        # Encode categorical features
        df_encoded = df_clean.copy()
        for col in feature_columns:
            if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
                # Simple label encoding for categorical
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        target = pd.to_numeric(df_encoded[target_column], errors='coerce')
        features = df_encoded[feature_columns].apply(pd.to_numeric, errors='coerce')
        
        drivers = []
        
        if method == 'correlation':
            for col in feature_columns:
                try:
                    corr = features[col].corr(target)
                    if not pd.isna(corr):
                        drivers.append({
                            "feature": col,
                            "correlation": round(corr, 4),
                            "impact": "positive" if corr > 0 else "negative",
                            "strength": "strong" if abs(corr) > 0.5 else ("moderate" if abs(corr) > 0.3 else "weak")
                        })
                except:
                    continue
        
        elif method == 'importance':
            try:
                # Use Random Forest for feature importance
                X = features.fillna(0)
                y = target.fillna(target.mean())
                
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X, y)
                
                importances = model.feature_importances_
                for col, imp in zip(feature_columns, importances):
                    drivers.append({
                        "feature": col,
                        "importance": round(imp, 4),
                        "rank": None  # Will be set after sorting
                    })
            except:
                # Fallback to correlation
                return await self.driver_analysis(df, target_column, feature_columns, 'correlation')
        
        # Sort by absolute value
        if method == 'correlation':
            drivers.sort(key=lambda x: abs(x['correlation']), reverse=True)
        else:
            drivers.sort(key=lambda x: x['importance'], reverse=True)
            for i, d in enumerate(drivers):
                d['rank'] = i + 1
        
        return {
            "target": target_column,
            "method": method,
            "drivers": drivers[:15],
            "top_driver": drivers[0] if drivers else None,
            "caveat": "Correlation does not imply causation. These are statistical associations only."
        }


# Singleton instance
analytics_engine = AnalyticsEngine()
