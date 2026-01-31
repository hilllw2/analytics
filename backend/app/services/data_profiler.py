"""
Data Profiling Service
Generates automatic data health reports and insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from dataclasses import dataclass, asdict
from datetime import datetime

from app.core.config import settings
from app.core.session_manager import DatasetInfo


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    name: str
    dtype: str
    inferred_type: str
    total_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    is_constant: bool
    is_near_constant: bool
    has_mixed_types: bool
    sample_values: List[Any]
    
    # Numeric stats (if applicable)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outlier_count: Optional[int] = None
    outlier_pct: Optional[float] = None
    
    # Categorical stats (if applicable)
    top_values: Optional[List[Dict[str, Any]]] = None
    
    # Date stats (if applicable)
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[int] = None


@dataclass
class DatasetHealthReport:
    """Complete health report for a dataset."""
    generated_at: str
    dataset_name: str
    row_count: int
    column_count: int
    memory_usage_mb: float
    
    # Overall quality metrics
    total_missing_cells: int
    total_missing_pct: float
    duplicate_row_count: int
    duplicate_row_pct: float
    
    # Column classifications
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    constant_columns: List[str]
    high_missing_columns: List[str]  # >50% missing
    
    # Column profiles
    column_profiles: Dict[str, Dict]
    
    # Correlation matrix (numeric columns only)
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    high_correlations: Optional[List[Dict[str, Any]]] = None
    
    # Missingness patterns
    missingness_heatmap: Optional[List[Dict[str, Any]]] = None
    
    # Warnings
    warnings: List[str] = None


class DataProfiler:
    """
    Service for profiling datasets and generating health reports.
    """
    
    def __init__(self):
        pass
    
    async def generate_health_report(
        self,
        dataset: DatasetInfo,
        include_correlations: bool = True
    ) -> DatasetHealthReport:
        """Generate a comprehensive health report for a dataset."""
        df = dataset.df
        
        # Basic counts
        row_count = len(df)
        column_count = len(df.columns)
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Missing values
        total_cells = row_count * column_count
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / row_count * 100) if row_count > 0 else 0
        
        # Profile each column
        column_profiles = {}
        constant_columns = []
        high_missing_columns = []
        
        for col in df.columns:
            profile = await self._profile_column(df[col], dataset.inferred_types.get(col, 'unknown'))
            column_profiles[col] = asdict(profile)
            
            if profile.is_constant:
                constant_columns.append(col)
            if profile.missing_pct > 50:
                high_missing_columns.append(col)
        
        # Correlation matrix
        correlation_matrix = None
        high_correlations = None
        if include_correlations and dataset.numeric_columns:
            correlation_matrix, high_correlations = await self._compute_correlations(
                df, dataset.numeric_columns
            )
        
        # Missingness heatmap data
        missingness_heatmap = await self._compute_missingness_patterns(df)
        
        # Generate warnings
        warnings = self._generate_warnings(
            df, column_profiles, constant_columns, high_missing_columns, duplicate_pct
        )
        
        return DatasetHealthReport(
            generated_at=datetime.now().isoformat(),
            dataset_name=dataset.name,
            row_count=row_count,
            column_count=column_count,
            memory_usage_mb=round(memory_mb, 2),
            total_missing_cells=int(missing_cells),
            total_missing_pct=round(missing_pct, 2),
            duplicate_row_count=int(duplicate_count),
            duplicate_row_pct=round(duplicate_pct, 2),
            numeric_columns=dataset.numeric_columns,
            categorical_columns=dataset.categorical_columns,
            datetime_columns=dataset.date_columns,
            constant_columns=constant_columns,
            high_missing_columns=high_missing_columns,
            column_profiles=column_profiles,
            correlation_matrix=correlation_matrix,
            high_correlations=high_correlations,
            missingness_heatmap=missingness_heatmap,
            warnings=warnings
        )
    
    async def _profile_column(self, series: pd.Series, inferred_type: str) -> ColumnProfile:
        """Profile a single column."""
        try:
            total = len(series)
            missing = int(series.isna().sum())
            missing_pct = (missing / total * 100) if total > 0 else 0
            
            non_null = series.dropna()
            unique = int(series.nunique())
            unique_pct = (unique / total * 100) if total > 0 else 0
            
            # Check for constant/near-constant
            is_constant = unique <= 1
            is_near_constant = False
            if not is_constant and len(non_null) > 0:
                try:
                    top_freq = non_null.value_counts().iloc[0] / len(non_null)
                    is_near_constant = top_freq >= settings.NEAR_CONSTANT_THRESHOLD
                except:
                    pass
            
            # Check for mixed types
            has_mixed_types = self._check_mixed_types(non_null)
            
            # Sample values - convert to basic types
            sample_values = []
            if len(non_null) > 0:
                for val in non_null.head(5):
                    if pd.isna(val):
                        continue
                    if hasattr(val, 'isoformat'):
                        sample_values.append(val.isoformat())
                    elif hasattr(val, 'item'):
                        sample_values.append(val.item())
                    else:
                        sample_values.append(str(val))
            
            profile = ColumnProfile(
                name=str(series.name),
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                total_count=total,
                missing_count=missing,
                missing_pct=round(float(missing_pct), 2),
                unique_count=unique,
                unique_pct=round(float(unique_pct), 2),
                is_constant=is_constant,
                is_near_constant=is_near_constant,
                has_mixed_types=has_mixed_types,
                sample_values=sample_values
            )
            
            # Add type-specific stats
            if inferred_type in ['integer', 'float'] and len(non_null) > 0:
                profile = self._add_numeric_stats(profile, non_null)
            elif inferred_type == 'datetime' and len(non_null) > 0:
                profile = self._add_date_stats(profile, non_null)
            elif inferred_type in ['string', 'category'] and len(non_null) > 0:
                profile = self._add_categorical_stats(profile, non_null)
            
            return profile
        except Exception as e:
            # Return minimal profile on error
            return ColumnProfile(
                name=str(series.name),
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                total_count=len(series),
                missing_count=0,
                missing_pct=0.0,
                unique_count=0,
                unique_pct=0.0,
                is_constant=False,
                is_near_constant=False,
                has_mixed_types=False,
                sample_values=[]
            )
    
    def _check_mixed_types(self, series: pd.Series) -> bool:
        """Check if a column has mixed types (e.g., '12', '12.0', 'N/A')."""
        if len(series) == 0:
            return False
        
        sample = series.head(1000).astype(str)
        
        # Check for numeric-like values mixed with text
        numeric_pattern = sample.str.match(r'^-?\d+\.?\d*$', na=False)
        has_numeric = numeric_pattern.any()
        has_non_numeric = (~numeric_pattern).any()
        
        return has_numeric and has_non_numeric
    
    def _add_numeric_stats(self, profile: ColumnProfile, series: pd.Series) -> ColumnProfile:
        """Add numeric statistics to profile."""
        try:
            numeric = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric) == 0:
                return profile
            
            profile.mean = round(float(numeric.mean()), 4)
            profile.std = round(float(numeric.std()), 4)
            profile.min = round(float(numeric.min()), 4)
            profile.max = round(float(numeric.max()), 4)
            profile.median = round(float(numeric.median()), 4)
            profile.q1 = round(float(numeric.quantile(0.25)), 4)
            profile.q3 = round(float(numeric.quantile(0.75)), 4)
            
            # Skewness and kurtosis
            if len(numeric) > 2:
                profile.skewness = round(float(stats.skew(numeric)), 4)
                profile.kurtosis = round(float(stats.kurtosis(numeric)), 4)
            
            # Outliers using IQR
            iqr = profile.q3 - profile.q1
            lower = profile.q1 - (settings.OUTLIER_IQR_MULTIPLIER * iqr)
            upper = profile.q3 + (settings.OUTLIER_IQR_MULTIPLIER * iqr)
            outliers = ((numeric < lower) | (numeric > upper)).sum()
            profile.outlier_count = int(outliers)
            profile.outlier_pct = round(outliers / len(numeric) * 100, 2)
            
        except Exception:
            pass
        
        return profile
    
    def _add_date_stats(self, profile: ColumnProfile, series: pd.Series) -> ColumnProfile:
        """Add datetime statistics to profile."""
        try:
            dates = pd.to_datetime(series, errors='coerce').dropna()
            if len(dates) == 0:
                return profile
            
            profile.min_date = dates.min().isoformat()
            profile.max_date = dates.max().isoformat()
            profile.date_range_days = (dates.max() - dates.min()).days
            
        except Exception:
            pass
        
        return profile
    
    def _add_categorical_stats(self, profile: ColumnProfile, series: pd.Series) -> ColumnProfile:
        """Add categorical statistics to profile."""
        try:
            value_counts = series.value_counts().head(10)
            total = len(series)
            
            profile.top_values = [
                {
                    "value": str(val),
                    "count": int(count),
                    "percentage": round(count / total * 100, 2)
                }
                for val, count in value_counts.items()
            ]
        except Exception:
            pass
        
        return profile
    
    async def _compute_correlations(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str]
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        """Compute correlation matrix and identify high correlations."""
        if len(numeric_columns) < 2:
            return None, None
        
        # Select numeric columns and compute correlation
        numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        corr_matrix = numeric_df.corr()
        
        # Convert to dict format
        corr_dict = {}
        for col in corr_matrix.columns:
            corr_dict[col] = {
                other: round(corr_matrix.loc[col, other], 4)
                for other in corr_matrix.columns
            }
        
        # Find high correlations
        high_corrs = []
        seen_pairs = set()
        
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                pair = tuple(sorted([col1, col2]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) >= settings.CORRELATION_THRESHOLD:
                        high_corrs.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(corr_val, 4),
                            "strength": "strong" if abs(corr_val) >= 0.7 else "moderate"
                        })
        
        # Sort by absolute correlation
        high_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return corr_dict, high_corrs[:20]  # Top 20
    
    async def _compute_missingness_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Compute missingness patterns for heatmap."""
        missing_pct = df.isna().mean() * 100
        
        return [
            {
                "column": col,
                "missing_pct": round(pct, 2),
                "category": "high" if pct > 50 else ("medium" if pct > 10 else "low")
            }
            for col, pct in missing_pct.items()
        ]
    
    def _generate_warnings(
        self,
        df: pd.DataFrame,
        profiles: Dict[str, Dict],
        constant_cols: List[str],
        high_missing_cols: List[str],
        duplicate_pct: float
    ) -> List[str]:
        """Generate data quality warnings."""
        warnings = []
        
        if constant_cols:
            warnings.append(f"⚠️ {len(constant_cols)} column(s) have constant values: {', '.join(constant_cols[:5])}")
        
        if high_missing_cols:
            warnings.append(f"⚠️ {len(high_missing_cols)} column(s) have >50% missing values: {', '.join(high_missing_cols[:5])}")
        
        if duplicate_pct > 10:
            warnings.append(f"⚠️ {duplicate_pct:.1f}% of rows are duplicates")
        
        # Check for mixed types
        mixed_type_cols = [
            name for name, profile in profiles.items()
            if profile.get("has_mixed_types", False)
        ]
        if mixed_type_cols:
            warnings.append(f"⚠️ {len(mixed_type_cols)} column(s) have mixed types: {', '.join(mixed_type_cols[:5])}")
        
        # Check for high outlier columns
        high_outlier_cols = [
            name for name, profile in profiles.items()
            if profile.get("outlier_pct", 0) > 5
        ]
        if high_outlier_cols:
            warnings.append(f"ℹ️ {len(high_outlier_cols)} numeric column(s) have >5% outliers: {', '.join(high_outlier_cols[:5])}")
        
        return warnings


# Singleton instance
data_profiler = DataProfiler()
