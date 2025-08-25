from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, base64, mimetypes, json
from openai import OpenAI
import httpx  # pour gérer proprement les timeouts réseau

app = FastAPI(title="Garantix Extract API")

# CORS (utile pour l'app iOS et tests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——— Réglages sécurité/fiabilité ———
MAX_UPLOAD_BYTES = 6 * 1024 * 1024  # ~6 Mo (évite 502 sur gros PDF/images)
OPENAI_MODEL = "gpt-4.1-mini"

# Client OpenAI avec timeout raisonnable pour Render
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=45.0,   # secondes (total request)
    max_retries=1
)

# ——— Endpoints de base ———
@app.get("/")
def hello():
    return {"status": "ok", "message": "Garantix backend en ligne"}

@app.get("/health")
def health():
    return {"ok": True}

# ——— Utilitaire ———
def to_data_url(content: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# ——— Extraction facture ———
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        # 1) Lecture et vérifs
        blob = await file.read()
        if not blob:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(blob) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(blob)//1024} KB). Try a file under {MAX_UPLOAD_BYTES//1024} KB."
            )

        api_key = os.getenv("OPENAI_API_KEY") or ""
        if not api_key.startswith(("sk-", "sk-proj-")):
            print("❌ OPENAI_API_KEY manquant/invalide")
            raise HTTPException(status_code=500, detail="Server misconfigured (OPENAI_API_KEY)")

        data_url = to_data_url(blob, file.filename)

        # 2) Prompt + appel OpenAI (format JSON strict)
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
            model=OPENAI_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_image", "image_url": data_url}
                ]
            }],
            response_format={"type": "json_object"}
        )

        # 3) Parsing sécurisé
        text = resp.output[0].content[0].text  # string JSON
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"raw": text}  # renvoi brut si le JSON est imparfait

        return {"data": parsed}

    except HTTPException:
        raise
    except (httpx.TimeoutException, httpx.ReadTimeout) as e:
        print("⏱️ Timeout OpenAI:", repr(e))
        raise HTTPException(status_code=504, detail="Upstream timeout (try a smaller/clearer file)")
    except Exception as e:
        # Log côté serveur (visible dans Render → Logs) + message clair client
        print("❌ OpenAI/Server error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Upstream error: {str(e)}")
