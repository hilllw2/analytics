"""
LLM Service
Integrates with Google Gemini 2.5 Flash for natural language processing.
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
    Service for LLM-powered natural language understanding and code generation.
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
    
    async def process_query(
        self,
        query: str,
        session: Session,
        regenerate: bool = False,
        style: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a natural language query."""
        self._ensure_initialized()
        
        if not self.model:
            return {
                "type": "error",
                "response": "LLM not configured. Please check your GEMINI_API_KEY.",
                "code": None,
                "chart_spec": None,
                "methodology": None,
                "warnings": ["LLM service not available"]
            }
        
        context = self._build_context(session)
        
        prompt = f"""You are a data analyst assistant. Analyze the user's data and answer their question.

{context}

USER QUESTION: {query}

CRITICAL RULES:
1. ONLY use column names that EXIST in the dataset (listed above)
2. DO NOT make up or assume columns that don't exist
3. The dataframe variable is 'df'
4. Store your final result in a variable called 'result'
5. For charts, use columns from your 'result' dataframe (the output of your code)

RESPONSE FORMAT - Return ONLY this JSON (no markdown, no extra text):
{{
    "response": "Clear explanation of findings in plain English",
    "code": "Python pandas code using df, result must be a DataFrame or Series. Example: result = df.groupby('Column1')['Column2'].sum().reset_index()",
    "chart_spec": {{
        "type": "bar|line|pie|scatter|histogram|box",
        "x": "column from result dataframe",
        "y": "column from result dataframe", 
        "title": "Descriptive title"
    }},
    "methodology": "How you approached this analysis"
}}

EXAMPLES:
- If user asks "show sales by region": 
  code: result = df.groupby('Region')['Sales'].sum().reset_index()
  chart_spec: {{"type": "bar", "x": "Region", "y": "Sales", "title": "Sales by Region"}}

- If user asks "distribution of prices":
  code: result = df[['Price']].dropna()
  chart_spec: {{"type": "histogram", "x": "Price", "y": null, "title": "Price Distribution"}}

- If user asks "trend over time":
  code: result = df.groupby('Date')['Amount'].sum().reset_index()
  chart_spec: {{"type": "line", "x": "Date", "y": "Amount", "title": "Amount Over Time"}}

Now analyze the user's question and respond with valid JSON only:"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            print(f"📝 LLM Raw Response: {response_text[:500]}...")
            
            # Parse the response
            parsed = self._parse_response(response_text)
            
            return {
                "type": "text",
                "response": parsed.get("response", "Analysis complete."),
                "code": parsed.get("code"),
                "chart_spec": parsed.get("chart_spec"),
                "methodology": parsed.get("methodology"),
                "warnings": []
            }
            
        except Exception as e:
            error_msg = f"LLM Error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return {
                "type": "error",
                "response": error_msg,
                "code": None,
                "chart_spec": None,
                "methodology": None,
                "warnings": [error_msg]
            }
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM response, handling various formats."""
        # Clean up the text
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            # Remove ```json or ``` at start
            text = re.sub(r'^```(?:json)?\s*', '', text)
            # Remove ``` at end
            text = re.sub(r'\s*```$', '', text)
        
        # Try to parse as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return the text as response
        return {
            "response": text,
            "code": None,
            "chart_spec": None,
            "methodology": None
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
