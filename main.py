from fastapi import FastAPI

app = FastAPI(title="Garantix (sanity)")

@app.get("/")
def root():
    return {"status": "ok", "message": "sanity boot"}
