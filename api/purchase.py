from fastapi import APIRouter, HTTPException
import httpx

SPRING_BASE_URL = "http://localhost:8080"

router = APIRouter(prefix="/fastApi", tags=["api"])

@router.get("/purchase/{purchase_id}")
async def get_purchase(purchase_id: int):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{SPRING_BASE_URL}/api/purchase/{purchase_id}",
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"spring server error: {e!r}")

    return {
        "status_code": resp.status_code,
        "body": resp.json() if resp.content else None,
    }
