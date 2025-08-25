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
    try:
        # --- Lecture & vérifs basiques ---
        blob = await file.read()
        if not blob:
            raise HTTPException(status_code=400, detail="Empty file")

        api_key = os.getenv("OPENAI_API_KEY") or ""
        if not api_key or not api_key.startswith(("sk-", "sk-proj-")):
            # on log côté serveur
            print("❌ OPENAI_API_KEY manquant ou invalide")
            raise HTTPException(status_code=500, detail="Server misconfigured (OPENAI_API_KEY)")

        # --- Encodage image/PDF ---
        data_url = to_data_url(blob, file.filename)

        # --- Prompt & appel OpenAI ---
        prompt = (
            "Analyse cette facture française. Retourne UNIQUEMENT un JSON avec: "
            "seller{name,legal_name,vat_id,siret,address,email,website}; "
            "purchase_date (YYYY-MM-DD); currency; "
            "totals{subtotal_ht,tva_rate,tva_amount,total_ttc}; "
            "items[{product_name,sku,qty,unit_price,line_total,product_photo_url,"
            "warranty{duration_months,notes},confidence}]; "
            "invoice_number; order_number; payment_method; confidence. "
            "Si une info est absente, mets null. N'invente rien."
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

        text = resp.output[0].content[0].text  # string JSON
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"raw": text}

        return {"data": parsed}

    except HTTPException:
        # On laisse passer tel quel
        raise
    except Exception as e:
        # On log côté serveur (visible dans Render Logs) ET on retourne un message clair
        print("❌ OpenAI/Server error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Upstream error: {str(e)}")
