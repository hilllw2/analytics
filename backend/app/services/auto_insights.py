"""
Auto Insights Engine
Generates human-readable insights from data automatically.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.config import settings
from app.core.session_manager import DatasetInfo


@dataclass
class Insight:
    """A single insight about the data."""
    id: str
    category: str  # 'distribution', 'trend', 'anomaly', 'segment', 'relationship'
    title: str
    description: str
    importance: str  # 'high', 'medium', 'low'
    data: Optional[Dict[str, Any]] = None
    suggested_action: Optional[str] = None


class AutoInsightsEngine:
    """
    Automatically generates human-readable insights from data.
    """
    
    def __init__(self):
        self.insight_counter = 0
    
    async def generate_insights(
        self,
        dataset: DatasetInfo,
        max_insights: int = 15
    ) -> List[Dict[str, Any]]:
        """Generate automatic insights for a dataset."""
        insights = []
        df = dataset.df
        
        # 1. Overview insights
        insights.extend(await self._generate_overview_insights(df, dataset))
        
        # 2. Distribution insights for numeric columns
        for col in dataset.numeric_columns[:5]:  # Top 5 numeric
            insights.extend(await self._generate_distribution_insights(df, col))
        
        # 3. Segment insights for categorical columns
        if dataset.categorical_columns and dataset.numeric_columns:
            insights.extend(await self._generate_segment_insights(
                df, dataset.categorical_columns, dataset.numeric_columns
            ))
        
        # 4. Time-based insights if date columns exist
        if dataset.date_columns and dataset.numeric_columns:
            insights.extend(await self._generate_time_insights(
                df, dataset.date_columns[0], dataset.numeric_columns
            ))
        
        # 5. Anomaly insights
        insights.extend(await self._generate_anomaly_insights(df, dataset.numeric_columns))
        
        # 6. Relationship insights
        if len(dataset.numeric_columns) >= 2:
            insights.extend(await self._generate_relationship_insights(
                df, dataset.numeric_columns
            ))
        
        # Sort by importance and limit
        importance_order = {'high': 0, 'medium': 1, 'low': 2}
        insights.sort(key=lambda x: importance_order.get(x.importance, 2))
        
        return [self._insight_to_dict(i) for i in insights[:max_insights]]
    
    def _insight_to_dict(self, insight: Insight) -> Dict[str, Any]:
        """Convert Insight to dictionary."""
        return {
            "id": insight.id,
            "category": insight.category,
            "title": insight.title,
            "description": insight.description,
            "importance": insight.importance,
            "data": insight.data,
            "suggested_action": insight.suggested_action
        }
    
    def _create_insight(
        self,
        category: str,
        title: str,
        description: str,
        importance: str = "medium",
        data: Optional[Dict] = None,
        suggested_action: Optional[str] = None
    ) -> Insight:
        """Create an insight with auto-generated ID."""
        self.insight_counter += 1
        return Insight(
            id=f"insight_{self.insight_counter}",
            category=category,
            title=title,
            description=description,
            importance=importance,
            data=data,
            suggested_action=suggested_action
        )
    
    async def _generate_overview_insights(
        self,
        df: pd.DataFrame,
        dataset: DatasetInfo
    ) -> List[Insight]:
        """Generate overview insights about the dataset."""
        insights = []
        
        # Data size insight
        insights.append(self._create_insight(
            category="overview",
            title="Dataset Overview",
            description=f"Your dataset has {len(df):,} rows and {len(df.columns)} columns. "
                       f"There are {len(dataset.numeric_columns)} numeric, "
                       f"{len(dataset.categorical_columns)} categorical, and "
                       f"{len(dataset.date_columns)} date columns.",
            importance="high"
        ))
        
        # Missing data insight
        missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > 0:
            severity = "high" if missing_pct > 20 else ("medium" if missing_pct > 5 else "low")
            insights.append(self._create_insight(
                category="quality",
                title="Missing Data Summary",
                description=f"{missing_pct:.1f}% of cells contain missing values. "
                           f"Columns with most missing: {', '.join(df.isna().sum().nlargest(3).index.tolist())}.",
                importance=severity,
                suggested_action="Consider imputing missing values or excluding high-missing columns from analysis."
            ))
        
        return insights
    
    async def _generate_distribution_insights(
        self,
        df: pd.DataFrame,
        column: str
    ) -> List[Insight]:
        """Generate insights about numeric column distributions."""
        insights = []
        
        try:
            numeric = pd.to_numeric(df[column], errors='coerce').dropna()
            if len(numeric) < 10:
                return insights
            
            mean_val = numeric.mean()
            median_val = numeric.median()
            std_val = numeric.std()
            
            # Skewness insight
            skewness = stats.skew(numeric)
            if abs(skewness) > 1:
                direction = "right (positive)" if skewness > 0 else "left (negative)"
                insights.append(self._create_insight(
                    category="distribution",
                    title=f"Skewed Distribution: {column}",
                    description=f"'{column}' is highly skewed to the {direction} "
                               f"(skewness: {skewness:.2f}). The mean ({mean_val:,.2f}) differs "
                               f"significantly from the median ({median_val:,.2f}).",
                    importance="medium",
                    data={"skewness": round(skewness, 3), "mean": round(mean_val, 2), "median": round(median_val, 2)},
                    suggested_action="Consider using median instead of mean for central tendency."
                ))
            
            # Top contributor insight (if values sum to a meaningful total)
            if (numeric >= 0).all():
                total = numeric.sum()
                top_vals = numeric.nlargest(int(len(numeric) * 0.2))  # Top 20%
                top_contribution = top_vals.sum() / total * 100 if total > 0 else 0
                
                if top_contribution > 70:
                    insights.append(self._create_insight(
                        category="concentration",
                        title=f"High Concentration: {column}",
                        description=f"The top 20% of values in '{column}' account for "
                                   f"{top_contribution:.1f}% of the total. This suggests high concentration.",
                        importance="medium",
                        data={"top_20_pct_contribution": round(top_contribution, 1)}
                    ))
            
        except Exception:
            pass
        
        return insights
    
    async def _generate_segment_insights(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        numeric_cols: List[str]
    ) -> List[Insight]:
        """Generate segment-based insights."""
        insights = []
        
        # Pick first numeric as primary metric
        metric_col = numeric_cols[0]
        
        for cat_col in categorical_cols[:3]:  # Top 3 categorical
            try:
                # Check cardinality
                if df[cat_col].nunique() > 20:
                    continue
                
                # Compute segment totals
                segment_totals = df.groupby(cat_col)[metric_col].agg(['sum', 'mean', 'count'])
                segment_totals = segment_totals[segment_totals['count'] >= 5]  # Min 5 rows
                
                if len(segment_totals) < 2:
                    continue
                
                total = segment_totals['sum'].sum()
                if total <= 0:
                    continue
                
                # Find top contributor
                segment_totals['pct'] = segment_totals['sum'] / total * 100
                top_segment = segment_totals['pct'].idxmax()
                top_pct = segment_totals.loc[top_segment, 'pct']
                
                if top_pct > 30:
                    insights.append(self._create_insight(
                        category="segment",
                        title=f"Top Segment by {metric_col}",
                        description=f"'{top_segment}' in '{cat_col}' drives {top_pct:.1f}% of total {metric_col} "
                                   f"({segment_totals.loc[top_segment, 'sum']:,.2f} out of {total:,.2f}).",
                        importance="high",
                        data={
                            "segment_column": cat_col,
                            "top_segment": str(top_segment),
                            "metric": metric_col,
                            "percentage": round(top_pct, 1)
                        },
                        suggested_action=f"Investigate what makes '{top_segment}' the largest contributor."
                    ))
                
                # Find segment with highest average
                top_avg_segment = segment_totals['mean'].idxmax()
                top_avg = segment_totals.loc[top_avg_segment, 'mean']
                overall_avg = df[metric_col].mean()
                
                if top_avg > overall_avg * 1.5:
                    pct_above = (top_avg / overall_avg - 1) * 100
                    insights.append(self._create_insight(
                        category="segment",
                        title=f"High-Performing Segment",
                        description=f"'{top_avg_segment}' has {pct_above:.0f}% higher average {metric_col} "
                                   f"({top_avg:,.2f}) than the overall average ({overall_avg:,.2f}).",
                        importance="medium",
                        data={
                            "segment": str(top_avg_segment),
                            "segment_avg": round(top_avg, 2),
                            "overall_avg": round(overall_avg, 2)
                        }
                    ))
                
            except Exception:
                continue
        
        return insights
    
    async def _generate_time_insights(
        self,
        df: pd.DataFrame,
        date_col: str,
        numeric_cols: List[str]
    ) -> List[Insight]:
        """Generate time-based insights."""
        insights = []
        
        try:
            # Parse dates
            dates = pd.to_datetime(df[date_col], errors='coerce')
            df_time = df.copy()
            df_time['_date'] = dates
            df_time = df_time.dropna(subset=['_date'])
            
            if len(df_time) < 10:
                return insights
            
            metric_col = numeric_cols[0]
            
            # Create time series
            df_time = df_time.sort_values('_date')
            
            # Monthly aggregation
            df_time['_month'] = df_time['_date'].dt.to_period('M')
            monthly = df_time.groupby('_month')[metric_col].sum()
            
            if len(monthly) >= 2:
                # Trend direction
                first_half = monthly.iloc[:len(monthly)//2].mean()
                second_half = monthly.iloc[len(monthly)//2:].mean()
                
                if first_half > 0:
                    change_pct = (second_half / first_half - 1) * 100
                    
                    if abs(change_pct) > 10:
                        direction = "increased" if change_pct > 0 else "decreased"
                        insights.append(self._create_insight(
                            category="trend",
                            title=f"Trend: {metric_col} Over Time",
                            description=f"{metric_col} has {direction} by {abs(change_pct):.1f}% "
                                       f"comparing the first half to the second half of the time period.",
                            importance="high",
                            data={
                                "first_half_avg": round(first_half, 2),
                                "second_half_avg": round(second_half, 2),
                                "change_pct": round(change_pct, 1)
                            }
                        ))
                
                # Peak period
                peak_month = monthly.idxmax()
                peak_value = monthly.max()
                avg_value = monthly.mean()
                
                if peak_value > avg_value * 1.5:
                    insights.append(self._create_insight(
                        category="trend",
                        title=f"Peak Period Identified",
                        description=f"The highest {metric_col} was in {peak_month} "
                                   f"({peak_value:,.2f}), which is {(peak_value/avg_value - 1)*100:.0f}% above average.",
                        importance="medium",
                        data={
                            "peak_period": str(peak_month),
                            "peak_value": round(peak_value, 2),
                            "average": round(avg_value, 2)
                        }
                    ))
            
            # Recent change
            if len(monthly) >= 3:
                last_month = monthly.iloc[-1]
                prev_month = monthly.iloc[-2]
                
                if prev_month > 0:
                    mom_change = (last_month / prev_month - 1) * 100
                    
                    if abs(mom_change) > 20:
                        direction = "up" if mom_change > 0 else "down"
                        insights.append(self._create_insight(
                            category="trend",
                            title=f"Recent Change: {metric_col}",
                            description=f"{metric_col} is {direction} {abs(mom_change):.0f}% in the most recent period "
                                       f"compared to the previous period.",
                            importance="high" if abs(mom_change) > 50 else "medium",
                            data={
                                "recent_period": round(last_month, 2),
                                "previous_period": round(prev_month, 2),
                                "change_pct": round(mom_change, 1)
                            },
                            suggested_action="Investigate what caused this significant change."
                        ))
        
        except Exception:
            pass
        
        return insights
    
    async def _generate_anomaly_insights(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> List[Insight]:
        """Generate anomaly-related insights."""
        insights = []
        
        for col in numeric_cols[:5]:
            try:
                numeric = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric) < 20:
                    continue
                
                # IQR-based outliers
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = q3 - q1
                
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                outliers = numeric[(numeric < lower) | (numeric > upper)]
                outlier_pct = len(outliers) / len(numeric) * 100
                
                if outlier_pct > 3:
                    extreme_high = outliers[outliers > upper]
                    extreme_low = outliers[outliers < lower]
                    
                    desc = f"'{col}' has {len(outliers)} outliers ({outlier_pct:.1f}% of values). "
                    if len(extreme_high) > 0:
                        desc += f"High outliers: max is {extreme_high.max():,.2f} (normal range up to {upper:,.2f}). "
                    if len(extreme_low) > 0:
                        desc += f"Low outliers: min is {extreme_low.min():,.2f} (normal range from {lower:,.2f})."
                    
                    insights.append(self._create_insight(
                        category="anomaly",
                        title=f"Outliers Detected: {col}",
                        description=desc,
                        importance="medium",
                        data={
                            "outlier_count": len(outliers),
                            "outlier_pct": round(outlier_pct, 1),
                            "lower_bound": round(lower, 2),
                            "upper_bound": round(upper, 2)
                        },
                        suggested_action="Review outliers to determine if they are errors or legitimate extreme values."
                    ))
                
            except Exception:
                continue
        
        return insights
    
    async def _generate_relationship_insights(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> List[Insight]:
        """Generate relationship insights between numeric columns."""
        insights = []
        
        if len(numeric_cols) < 2:
            return insights
        
        try:
            # Select subset of numeric columns
            cols_to_check = numeric_cols[:6]
            numeric_df = df[cols_to_check].apply(pd.to_numeric, errors='coerce')
            
            # Compute correlations
            corr = numeric_df.corr()
            
            # Find strong correlations
            for i, col1 in enumerate(cols_to_check):
                for col2 in cols_to_check[i+1:]:
                    corr_val = corr.loc[col1, col2]
                    
                    if pd.isna(corr_val):
                        continue
                    
                    if abs(corr_val) >= 0.7:
                        direction = "positive" if corr_val > 0 else "negative"
                        insights.append(self._create_insight(
                            category="relationship",
                            title=f"Strong Correlation Found",
                            description=f"'{col1}' and '{col2}' have a strong {direction} correlation "
                                       f"(r = {corr_val:.2f}). As one increases, the other tends to "
                                       f"{'increase' if corr_val > 0 else 'decrease'}.",
                            importance="medium",
                            data={
                                "column1": col1,
                                "column2": col2,
                                "correlation": round(corr_val, 3)
                            }
                        ))
                        
                        if len(insights) >= 3:  # Limit relationship insights
                            return insights
        
        except Exception:
            pass
        
        return insights
    
    async def generate_suggested_questions(
        self,
        dataset: DatasetInfo
    ) -> List[Dict[str, str]]:
        """Generate suggested follow-up questions based on the data."""
        suggestions = []
        df = dataset.df
        
        # Basic questions
        suggestions.append({
            "text": "Summarize this dataset",
            "category": "overview"
        })
        
        # Numeric-based questions
        if dataset.numeric_columns:
            metric = dataset.numeric_columns[0]
            suggestions.append({
                "text": f"What is the distribution of {metric}?",
                "category": "distribution"
            })
            suggestions.append({
                "text": f"Find outliers in {metric}",
                "category": "anomaly"
            })
        
        # Segment questions
        if dataset.categorical_columns and dataset.numeric_columns:
            cat = dataset.categorical_columns[0]
            metric = dataset.numeric_columns[0]
            suggestions.append({
                "text": f"Compare {metric} by {cat}",
                "category": "segment"
            })
            suggestions.append({
                "text": f"Which {cat} has the highest {metric}?",
                "category": "ranking"
            })
        
        # Time-based questions
        if dataset.date_columns and dataset.numeric_columns:
            date_col = dataset.date_columns[0]
            metric = dataset.numeric_columns[0]
            suggestions.append({
                "text": f"Show {metric} trend over time",
                "category": "trend"
            })
            suggestions.append({
                "text": f"What changed most recently?",
                "category": "trend"
            })
        
        # Top N questions
        if dataset.numeric_columns:
            suggestions.append({
                "text": f"Show top 10 rows by {dataset.numeric_columns[0]}",
                "category": "ranking"
            })
        
        return suggestions[:8]


# Singleton instance
auto_insights = AutoInsightsEngine()
