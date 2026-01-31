"""
LLM Service
Two-agent flow: Agent 1 (analysis/insights) -> Agent 2 (code/chart).
Integrates with Google Gemini for natural language processing.
"""

import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
import re
import traceback

from app.core.config import settings
from app.core.session_manager import Session, DatasetInfo


class LLMService:
    """
    Two-agent service:
    - Agent 1: Generates insights, summary, and data overview (no code).
    - Agent 2: Generates code and chart_spec using Agent 1 output and exact column names.
    """
    
    def __init__(self):
        self.model = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Initialize the Gemini client if not already done."""
        if not self._initialized and settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model = genai.GenerativeModel(settings.LLM_MODEL)
                self._initialized = True
                print(f"✅ LLM initialized with model: {settings.LLM_MODEL}")
            except Exception as e:
                print(f"❌ LLM initialization failed: {e}")
                self.model = None
    
    def _build_data_overview(self, session: Session) -> str:
        """Build a short df.info()-style overview for the code agent."""
        if not session.active_dataset:
            return "No dataset loaded."
        ds = session.active_dataset
        df = ds.df
        lines = [
            f"Shape: {df.shape[0]} rows, {df.shape[1]} columns",
            "Columns (exact names to use in code):",
            ", ".join(f'"{c}"' for c in df.columns),
            "",
            "Dtypes: " + ", ".join(f"{c}: {df[c].dtype}" for c in df.columns[:15])
            + ("..." if len(df.columns) > 15 else ""),
        ]
        return "\n".join(lines)
    
    def _build_context(self, session: Session) -> str:
        """Build comprehensive context about the dataset."""
        if not session.active_dataset:
            return "No dataset loaded."
        
        ds = session.active_dataset
        df = ds.df
        
        # Build detailed column info with stats
        column_details = []
        for col in df.columns:
            col_type = ds.inferred_types.get(col, 'unknown')
            col_info = f"  • {col} ({col_type})"
            
            try:
                non_null = df[col].dropna()
                null_count = df[col].isna().sum()
                
                if col_type in ['integer', 'float']:
                    # Numeric stats
                    col_info += f"\n      Range: {non_null.min()} to {non_null.max()}"
                    col_info += f", Mean: {non_null.mean():.2f}"
                    col_info += f", Nulls: {null_count}"
                elif col_type == 'datetime':
                    # Date range
                    col_info += f"\n      Range: {non_null.min()} to {non_null.max()}"
                    col_info += f", Nulls: {null_count}"
                elif col_type in ['string', 'category']:
                    # Top values
                    unique = df[col].nunique()
                    top_vals = non_null.value_counts().head(3).index.tolist()
                    col_info += f"\n      Unique: {unique}, Top values: {top_vals}"
                    col_info += f", Nulls: {null_count}"
            except:
                pass
            
            column_details.append(col_info)
        
        # Get sample rows
        try:
            sample_str = df.head(5).to_string(max_colwidth=30)
        except:
            sample_str = "Unable to show sample"
        
        context = f"""
========== DATASET INFORMATION ==========
Name: {ds.name}
Total Rows: {ds.row_count:,}
Total Columns: {ds.column_count}

========== COLUMNS (USE THESE EXACT NAMES) ==========
{chr(10).join(column_details)}

========== COLUMN LISTS ==========
NUMERIC COLUMNS (for calculations/aggregations): {ds.numeric_columns if ds.numeric_columns else []}
DATE/TIME COLUMNS (for time analysis): {ds.date_columns if ds.date_columns else []}
CATEGORICAL COLUMNS (for grouping): {ds.categorical_columns if ds.categorical_columns else []}

========== SAMPLE DATA (first 5 rows) ==========
{sample_str}
==========================================
"""
        return context
    
    def _agent_analysis(self, query: str, context: str) -> Dict[str, Any]:
        """
        Agent 1: Generate insights and summary only. No code.
        Returns: { "insights": str, "summary": str, "data_overview": str }
        """
        prompt = f"""You are a data analyst. Your job is ONLY to analyze the dataset and produce insights and a summary. Do NOT generate any code.

{context}

USER REQUEST: {query}

Output ONLY valid JSON with these exact keys (no markdown, no code blocks, no extra text):
{{
    "insights": "2-4 bullet points in plain English about what the data shows and what the user might want to see. Use exact column names from the dataset.",
    "summary": "One short paragraph summarizing the dataset and how it relates to the user's question.",
    "data_overview": "One line: row count, column names (list them), and which columns are numeric/categorical."
}}

Return only the JSON object, nothing else."""

        resp = self.model.generate_content(prompt)
        text = resp.text.strip()
        parsed = self._parse_json_only(text)
        return {
            "insights": parsed.get("insights", ""),
            "summary": parsed.get("summary", ""),
            "data_overview": parsed.get("data_overview", ""),
        }

    def _agent_code(
        self,
        query: str,
        agent1_output: Dict[str, Any],
        column_names: List[str],
        data_overview: str,
    ) -> Dict[str, Any]:
        """
        Agent 2: Generate only code and chart_spec. Uses Agent 1 output and exact column names.
        Returns: { "code": str, "chart_spec": dict }
        """
        insights = agent1_output.get("insights", "")
        summary = agent1_output.get("summary", "")
        cols_str = ", ".join(f'"{c}"' for c in column_names)

        prompt = f"""You are a code generator. You receive the user's question, an analysis summary from another agent, and the EXACT column names. Your job is ONLY to output JSON with "code" and "chart_spec". No explanations.

USER REQUEST: {query}

ANALYSIS FROM PREVIOUS AGENT (for context only):
Insights: {insights}
Summary: {summary}

DATA OVERVIEW:
{data_overview}

EXACT COLUMN NAMES (use these and only these in your code): {cols_str}

RULES:
1. The dataframe variable is "df". Your final result must be in a variable named "result" (DataFrame or Series).
2. Use ONLY the column names listed above. If a column is "Total Revenue" or "Sales Rep", use that exact string.
3. For chart_spec use EXACT column names that will exist in "result" after your code runs (e.g. if you groupby and reset_index, use the resulting column names).
4. chart_spec: {{ "type": "bar|line|pie|scatter|histogram|box", "x": "exact column name", "y": "exact column name", "title": "Chart title" }}

Output ONLY valid JSON (no markdown, no ```):
{{
    "code": "Python pandas code. Example: result = df.groupby('Col1')['Col2'].sum().reset_index()",
    "chart_spec": {{ "type": "bar", "x": "Col1", "y": "Col2", "title": "Title" }}
}}"""

        resp = self.model.generate_content(prompt)
        text = resp.text.strip()
        parsed = self._parse_json_only(text)
        return {
            "code": parsed.get("code"),
            "chart_spec": parsed.get("chart_spec"),
        }

    def _parse_json_only(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM output; never return raw text as content."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    async def process_query(
        self,
        query: str,
        session: Session,
        regenerate: bool = False,
        style: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Two-agent flow: Agent 1 (insights/summary) -> Agent 2 (code/chart).
        Returns only formatted insights for display; code and chart_spec for execution.
        """
        self._ensure_initialized()

        if not self.model:
            return {
                "type": "error",
                "response": "LLM not configured. Please check your GEMINI_API_KEY.",
                "code": None,
                "chart_spec": None,
                "methodology": None,
                "warnings": ["LLM service not available"],
            }

        context = self._build_context(session)
        data_overview = self._build_data_overview(session)
        column_names = list(session.active_dataset.df.columns) if session.active_dataset else []

        try:
            # Agent 1: insights and summary only (never shown as raw JSON)
            agent1 = self._agent_analysis(query, context)
            insights = (agent1.get("insights") or "").strip()
            summary = (agent1.get("summary") or "").strip()
            # Build the text we show in chat (insights + summary only)
            response_parts = []
            if insights:
                response_parts.append(insights)
            if summary:
                response_parts.append(summary)
            display_response = "\n\n".join(response_parts) if response_parts else "Analysis complete."

            # Agent 2: code and chart_spec using Agent 1 output
            agent2 = self._agent_code(query, agent1, column_names, data_overview)
            code = agent2.get("code")
            chart_spec = agent2.get("chart_spec")

            return {
                "type": "text",
                "response": display_response,
                "code": code,
                "chart_spec": chart_spec,
                "methodology": None,
                "warnings": [],
            }
        except Exception as e:
            error_msg = str(e)
            print(f"❌ LLM Error: {error_msg}")
            traceback.print_exc()
            return {
                "type": "error",
                "response": f"Error: {error_msg}",
                "code": None,
                "chart_spec": None,
                "methodology": None,
                "warnings": [error_msg],
            }

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM response; avoid ever returning raw JSON as the displayed response."""
        parsed = self._parse_json_only(text)
        if parsed:
            # Ensure we never send raw JSON to the user
            resp_text = parsed.get("response") or parsed.get("insights") or parsed.get("summary")
            if isinstance(resp_text, str) and resp_text.strip():
                return {**parsed, "response": resp_text.strip()}
            if parsed.get("insights") or parsed.get("summary"):
                parts = [parsed.get("insights"), parsed.get("summary")]
                return {**parsed, "response": "\n\n".join(p for p in parts if p).strip() or "Analysis complete."}
        return {
            "response": "Analysis complete.",
            "code": None,
            "chart_spec": None,
            "methodology": None,
        }
    
    async def generate_simple_response(
        self,
        query: str,
        session: Session
    ) -> str:
        """Generate a simple text response without code."""
        self._ensure_initialized()
        
        if not self.model:
            return "LLM not available."
        
        context = self._build_context(session)
        
        prompt = f"""Based on this dataset:
{context}

Answer this question concisely: {query}"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"


# Singleton instance
llm_service = LLMService()
