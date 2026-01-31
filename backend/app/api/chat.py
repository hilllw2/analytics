"""
Chat API Routes
Natural language chat interface for data analysis.
"""

from fastapi import APIRouter, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json

from app.core.session_manager import session_manager, PinnedDefinition
from app.services.llm_service import llm_service
from app.services.query_executor import query_executor
from app.services.visualization import visualization_service
from datetime import datetime


router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message from user."""
    message: str
    regenerate: bool = False
    style: Optional[str] = None  # 'detailed', 'simple', 'technical'


class PinDefinitionRequest(BaseModel):
    """Request to pin a definition."""
    name: str
    formula: str
    description: Optional[str] = None


@router.post("/message")
async def send_message(
    request: ChatMessage,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Process a natural language message and return analysis results.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if not session.active_dataset:
        raise HTTPException(
            status_code=400, 
            detail="No dataset loaded. Please upload a file first."
        )
    
    # Add user message to history
    session.add_message("user", request.message)
    
    try:
        # Process with LLM
        llm_response = await llm_service.process_query(
            query=request.message,
            session=session,
            regenerate=request.regenerate,
            style=request.style
        )
        
        # Never show raw LLM/JSON in chat: use only the parsed response text (string or list -> string)
        raw_response = llm_response.get("response", "") or ""
        if isinstance(raw_response, list):
            response_text = "\n".join(str(x).strip() for x in raw_response if x is not None).strip()
        elif isinstance(raw_response, str):
            response_text = raw_response.strip()
        else:
            response_text = str(raw_response).strip()
        if response_text.startswith("{"):
            response_text = "Analysis complete. See data and chart below."
        result = {
            "type": llm_response.get("type", "text"),
            "response": response_text or "Analysis complete.",
            "methodology": llm_response.get("methodology"),
            "warnings": llm_response.get("warnings", []),
            "data": None,
            "chart": None
        }
        
        # Execute code if provided
        exec_result_data = None
        if llm_response.get("code"):
            exec_result = await query_executor.execute_query(
                llm_response["code"],
                session
            )
            
            if exec_result["success"]:
                result["data"] = exec_result["result"]
                result["data_type"] = exec_result["result_type"]
                result["row_count"] = exec_result.get("row_count")
                result["truncated"] = exec_result.get("truncated", False)
                exec_result_data = exec_result.get("raw_result")  # Keep raw result for charting
                
                # Debug logging
                if exec_result_data is not None:
                    print(f"📊 Code executed successfully. Result type: {type(exec_result_data).__name__}")
                    if hasattr(exec_result_data, 'columns'):
                        print(f"📊 Result columns: {list(exec_result_data.columns)}")
                    elif hasattr(exec_result_data, 'name'):
                        print(f"📊 Result Series name: {exec_result_data.name}")
                else:
                    print(f"⚠️ Code executed but raw_result is None")
                
                # Store transformation in context
                session.current_transformations.append(llm_response["code"])
            else:
                result["warnings"].append(f"Query execution failed: {exec_result['error']}")
        
        # Generate chart if specified
        if llm_response.get("chart_spec"):
            chart_result = await _generate_chart_from_spec(
                llm_response["chart_spec"],
                session,
                exec_result_data  # Pass the code result for charting
            )
            if chart_result:
                result["chart"] = chart_result
        
        # Add assistant response to history
        session.add_message("assistant", result["response"], {
            "type": result["type"],
            "has_data": result["data"] is not None,
            "has_chart": result["chart"] is not None
        })
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        session.add_message("assistant", error_msg, {"error": True})
        raise HTTPException(status_code=500, detail=error_msg)


def _resolve_chart_columns(df, x: Optional[str], y: Optional[str], chart_type: str):
    """
    Resolve x/y to actual DataFrame column names when LLM uses generic names (e.g. Quarter, Sales).
    Returns (x_col, y_col) or (None, None) if resolution fails.
    """
    import pandas as pd
    cols = list(df.columns)
    if not cols:
        return None, None

    def _numeric_cols():
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    def _non_numeric_cols():
        return [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    def _best_x():
        if x and x in cols:
            return x
        # Prefer columns that look like categories / dimension
        x_keywords = ["quarter", "date", "channel", "category", "name", "region", "type", "group", "profit by sales", "channel and quarter"]
        for kw in x_keywords:
            for c in cols:
                if kw in str(c).lower():
                    return c
        # Fallback: first non-numeric
        non_num = _non_numeric_cols()
        return non_num[0] if non_num else cols[0]

    def _best_y():
        if y and y in cols:
            return y
        y_keywords = ["sales", "total", "value", "amount", "sum", "unnamed", "revenue", "profit"]
        for kw in y_keywords:
            for c in cols:
                if kw in str(c).lower():
                    if pd.api.types.is_numeric_dtype(df[c]):
                        return c
        num = _numeric_cols()
        return num[0] if num else (cols[1] if len(cols) > 1 else cols[0])

    if chart_type == "histogram":
        num = _numeric_cols()
        x_col = (x if x in cols else None) or (y if y in cols else None) or (num[0] if num else cols[0])
        return x_col, None
    x_col = _best_x()
    y_col = _best_y()
    return x_col, y_col


async def _generate_chart_from_spec(
    chart_spec: Dict[str, Any],
    session,
    result_df = None
) -> Optional[Dict[str, Any]]:
    """Generate a chart from the LLM-provided specification."""
    import pandas as pd
    
    try:
        # Use the result DataFrame if provided, otherwise use the original
        if result_df is not None:
            if isinstance(result_df, pd.DataFrame):
                df = result_df
                print(f"📊 Using result DataFrame with columns: {list(df.columns)}")
            elif isinstance(result_df, pd.Series):
                # Convert Series to DataFrame
                df = result_df.reset_index()
                df.columns = [result_df.index.name or 'index', result_df.name or 'value']
                print(f"📊 Using result Series (converted to DataFrame) with columns: {list(df.columns)}")
            else:
                df = session.active_df
                print(f"📊 Result is {type(result_df)}, using original DataFrame with columns: {list(df.columns)}")
        else:
            df = session.active_df
            print(f"📊 No result provided, using original DataFrame with columns: {list(df.columns)}")
        
        chart_type = chart_spec.get("type", "bar")
        
        # Handle different key names from LLM (x/y or name/value or names/values)
        x = chart_spec.get("x") or chart_spec.get("name") or chart_spec.get("names")
        y = chart_spec.get("y") or chart_spec.get("value") or chart_spec.get("values")
        color = chart_spec.get("color")
        title = chart_spec.get("title", "Chart")
        
        # Resolve to actual column names when LLM uses generic names (Quarter, Sales, etc.)
        x_col, y_col = _resolve_chart_columns(df, x, y, chart_type)
        if x_col is None:
            print(f"📊 Could not resolve chart columns for x={x}, y={y}")
            return None
        if chart_type != "histogram" and y_col is None:
            y_col = x_col  # fallback for single-column
        
        print(f"📊 Generating chart: type={chart_type}, x={x_col}, y={y_col}, title={title}")
        print(f"📊 Available columns: {list(df.columns)}")
        
        chart_info = None
        
        # For pie charts, we need names and values columns
        if chart_type == "pie":
            names_col = x_col
            values_col = y_col
            if names_col and values_col and names_col in df.columns and values_col in df.columns:
                chart_info = await visualization_service.create_pie_chart(
                    df, names_col, values_col, title
                )
            else:
                print(f"❌ Pie chart: columns not found. Need {names_col}, {values_col}")
        elif chart_type == "line":
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                chart_info = await visualization_service.create_line_chart(df, x_col, y_col, color, title)
        elif chart_type == "bar":
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                chart_info = await visualization_service.create_bar_chart(df, x_col, y_col, color, title=title)
        elif chart_type == "scatter":
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                chart_info = await visualization_service.create_scatter_plot(df, x_col, y_col, color, title=title)
        elif chart_type == "histogram":
            col = x_col
            if col and col in df.columns:
                chart_info = await visualization_service.create_histogram(df, col, title=title)
        elif chart_type == "box":
            col = y_col or x_col
            if col and col in df.columns:
                chart_info = await visualization_service.create_box_plot(df, col, color, title)
        elif chart_type == "area":
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                chart_info = await visualization_service.create_area_chart(df, x_col, y_col, color, title)
        else:
            # Default to bar
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                chart_info = await visualization_service.create_bar_chart(df, x_col, y_col, color, title=title)
        
        if chart_info is None:
            print(f"❌ Chart generation failed - columns not found or invalid spec")
            return None
        
        # Store chart in session
        session.charts[chart_info.id] = chart_info
        
        print(f"✅ Chart created: {chart_info.id}")
        
        return {
            "id": chart_info.id,
            "type": chart_info.chart_type,
            "title": chart_info.title,
            "plotly_json": chart_info.plotly_json
        }
        
    except Exception as e:
        print(f"❌ Chart generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


@router.post("/regenerate")
async def regenerate_response(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Regenerate the last response with different approach.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    # Find the last user message
    last_user_msg = None
    for msg in reversed(session.messages):
        if msg.role == "user":
            last_user_msg = msg.content
            break
    
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No previous message to regenerate")
    
    # Remove the last assistant response
    if session.messages and session.messages[-1].role == "assistant":
        session.messages.pop()
    
    # Re-process with regenerate flag
    request = ChatMessage(message=last_user_msg, regenerate=True)
    return await send_message(request, session_id)


@router.post("/style/{style}")
async def apply_style(
    style: str,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Apply a style to the last response (detailed, simple, technical).
    """
    if style not in ['detailed', 'simple', 'technical']:
        raise HTTPException(status_code=400, detail="Invalid style. Use: detailed, simple, technical")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    # Find the last user message
    last_user_msg = None
    for msg in reversed(session.messages):
        if msg.role == "user":
            last_user_msg = msg.content
            break
    
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No previous message")
    
    # Re-process with style
    request = ChatMessage(message=last_user_msg, style=style)
    return await send_message(request, session_id)


@router.post("/pin-definition")
async def pin_definition(
    request: PinDefinitionRequest,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Pin a metric definition for reuse in the session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session.pinned_definitions[request.name] = PinnedDefinition(
        name=request.name,
        formula=request.formula,
        description=request.description,
        created_at=datetime.now()
    )
    
    return {
        "success": True,
        "message": f"Definition '{request.name}' pinned successfully",
        "definitions": {
            name: {"formula": d.formula, "description": d.description}
            for name, d in session.pinned_definitions.items()
        }
    }


@router.get("/definitions")
async def get_definitions(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Get all pinned definitions for the session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {
        "definitions": {
            name: {
                "formula": d.formula,
                "description": d.description,
                "created_at": d.created_at.isoformat()
            }
            for name, d in session.pinned_definitions.items()
        }
    }


@router.delete("/definitions/{name}")
async def delete_definition(
    name: str,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Delete a pinned definition.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if name in session.pinned_definitions:
        del session.pinned_definitions[name]
        return {"success": True, "message": f"Definition '{name}' removed"}
    else:
        raise HTTPException(status_code=404, detail=f"Definition '{name}' not found")


@router.get("/history")
async def get_history(
    session_id: str = Header(..., alias="X-Session-ID"),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get conversation history for the session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    messages = session.messages[-limit:]
    
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ],
        "total_messages": len(session.messages)
    }


@router.get("/suggestions")
async def get_suggestions(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Get suggested queries based on current dataset and context.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if not session.active_dataset:
        return {"suggestions": []}
    
    # Import here to avoid circular dependency
    from app.services.auto_insights import auto_insights
    
    suggestions = await auto_insights.generate_suggested_questions(session.active_dataset)
    
    return {"suggestions": suggestions}


@router.post("/execute-code")
async def execute_custom_code(
    code: str,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Execute custom pandas code (for advanced users).
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if not session.active_dataset:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    result = await query_executor.execute_query(code, session)
    
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])
