from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from textToSql import EcommerceTextToSQLAgent
import logging
from typing import Optional, Dict, Any, List

router = APIRouter(tags=["text-to-sql"])
agent = EcommerceTextToSQLAgent()


class TextToSQLRequest(BaseModel):
    query: str

class TextToSQLResponse(BaseModel):
    query: str
    result: Optional[Dict[str, Any]] = None 
    sql: str
    results: str
    rag_context: str = ""

@router.post("/execute")
async def execute_text_to_sql(request: TextToSQLRequest = Body(...)):
    try:
        agent = EcommerceTextToSQLAgent()
        result = agent.execute_query(request.query)
        return result
    except Exception as e:
        logging.error(f"Text to SQL error: {e}")
        return {"status": "error", "error": str(e)}
