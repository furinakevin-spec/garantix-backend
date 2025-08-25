from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, base64, mimetypes, json
from openai import OpenAI

app = FastAPI(title="Garantix Extract API")

# CORS (utile pour dev/app mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clé OpenAI depuis l'env Render (Settings → Environment → OPENAI_API_KEY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def hello():
    return {"status": "ok", "message": "Garantix backend en ligne 🎉"}

@app.get("/health")
def health():
    return {"ok": True}

def to_data_url(content: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # 1) Lire le fichier
    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty file")
    data_url = to_data_url(blob, file.filename)

    # 2) Prompt clair + format JSON strict
    prompt = (
        "Analyse cette facture française. Retourne UNIQUEMENT un JSON avec ces champs : "
        "seller{name,legal_name,vat_id,siret,address,email,website}; "
        "purchase_date (YYYY-MM-DD); currency; "
        "totals{subtotal_ht,tva_rate,tva_amount,total_ttc}; "
        "items[ {product_name,sku,qty,unit_price,line_total,product_photo_url,"
        "warranty{duration_months,notes},confidence} ]; "
        "invoice_number; order_number; payment_method; confidence. "
        "Si une info est absente, mets null. N'invente rien."
    )

    try:
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
            # en dernier recours, retourner le texte brut pour debug
            parsed = {"raw": text}

        return {"data": parsed}

    except Exception as e:
        # Propager une erreur lisible côté client
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")
