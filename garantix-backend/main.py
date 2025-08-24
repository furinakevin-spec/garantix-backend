
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os, base64, mimetypes, json
from openai import OpenAI

app = FastAPI(title="Garantix Extract API")

# Autoriser toutes les origines (utile pour tests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def hello():
    return {"status": "ok", "message": "Garantix backend en ligne 🎉"}

def to_data_url(content: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        return {"error": "empty file"}
    data_url = to_data_url(content, file.filename)

    prompt = (
        "Analyse cette facture (FR) et retourne UNIQUEMENT un JSON avec: "
        "seller{name,legal_name,vat_id,siret,address,email,website}, "
        "purchase_date (YYYY-MM-DD), currency, totals{subtotal_ht,tva_rate,tva_amount,total_ttc}, "
        "items[{product_name,sku,qty,unit_price,line_total,product_photo_url,warranty{duration_months,notes},confidence}], "
        "invoice_number, order_number, payment_method, confidence. "
        "N'invente pas si l'info est absente."
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_image", "image_url": data_url}
            ]
        }],
        response_format={"type": "json_object"}
    )

    # Le modèle renvoie une string JSON
    text = resp.output[0].content[0].text
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"raw": text}  # en dernier recours

    return {"data": parsed}
