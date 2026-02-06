from fastapi import FastAPI
from api import purchase
from api import text2sql   
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Ecommerce Text-to-SQL API")
app.include_router(purchase.router) 
app.include_router(text2sql.router, prefix="/fastApi")  

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "Ecommerce Text-to-SQL API 실행", "docs": "/docs"}
