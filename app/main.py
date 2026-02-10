from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.model import summarize
from app.self_built_model import summarize_self_built

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/summarize", response_class=HTMLResponse)
async def make_summary(request: Request, text: str = Form(...)):
    result = summarize(text)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# Route cho model tự build
@app.get("/self-built", response_class=HTMLResponse)
async def self_built_page(request: Request):
    return templates.TemplateResponse("self_built.html", {"request": request, "result": None})

@app.post("/self-built-summarize", response_class=HTMLResponse)
async def self_built_summary(request: Request, text: str = Form(...)):
    result = summarize_self_built(text)
    return templates.TemplateResponse("self_built.html", {"request": request, "result": result})