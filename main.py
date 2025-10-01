from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn

import os
from flow import handle_flow_step
from rag import retrieve, ensure_index

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class FlowRequest(BaseModel):
    step: int
    answer: Optional[str] = ""
    answers: Optional[dict] = None

class RAGRequest(BaseModel):
    query: str
    k: Optional[int] = 3

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    ensure_index()
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/flow/respond")
async def flow_respond(req: FlowRequest):
    answers = req.answers if req.answers is not None else {}
    return handle_flow_step(req.step, req.answer or "", answers)

@app.post("/api/rag/query")
async def rag_query(req: RAGRequest):
    results = retrieve(req.query, k=req.k)
    return {"query": req.query, "results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
