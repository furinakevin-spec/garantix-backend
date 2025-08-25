from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, base64, mimetypes, json
from openai import OpenAI
import httpx  # timeouts réseau

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
OPENAI_MODEL = "gpt-4.1-mini"       # pour Responses (vision support)
FALLBACK_MODEL = "gpt-4o-mini"      # pour Chat Completions (vision)

# Client OpenAI avec timeout raisonnable pour Render
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=45.0,   # secondes
    max_retries=1
)

# ——— Endpoints de base ———
@app.get("/")
def hello():
    return {"status": "ok", "message": "Garantix backend en ligne"}

@app.get("/health")
def health():
    return {"ok": True}

# ——— Utilitaires ———
def to_data_url(content: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def mock_payload():
    # ⚠️ Exemple fictif — adapte-le si besoin
    return {
        "data": {
            "seller": {"name": "TEXIPOOL SASU", "legal_name": None, "vat_id": "FR89502939432",
                       "siret": "50293943200014", "address": "Rue de l'océan 44390 Nort-sur-Erdre",
                       "email": "commandes@baches-piscines.com", "website": "https://www.baches-piscines.com"},
            "purchase_date": "2025-08-20",
            "currency": "EUR",
            "totals": {"subtotal_ht": 648.87, "tva_rate": 20, "tva_amount": 129.77, "total_ttc": 778.64},
            "items": [
                {"product_name": "Enrouleur piscine 3-5 m Alpha axe 80mm", "sku": "A280167",
                 "qty": 1, "unit_price": 190.0, "line_total": 190.0,
                 "product_photo_url": None, "warranty": {"duration_months": 24, "notes": "2 ans"},
                 "confidence": 0.93},
                {"product_name": "Bâche bulles 500µ Noir Energy Guard Geobubble", "sku": "BBNOIRNET",
                 "qty": 1, "unit_price": 530.96, "line_total": 530.96,
                 "product_photo_url": None, "warranty": {"duration_months": 48, "notes": "4 ans"},
                 "confidence": 0.88}
            ],
            "invoice_number": "290144",
            "order_number": "299258",
            "payment_method": "PayPal",
            "confidence": 0.92
        }
    }

def call_openai_vision_json(data_url: str, prompt: str) -> str:
    """
    Tente d'abord l'API Responses (si dispo), sinon fallback Chat Completions (vision).
    Retourne une string JSON (ou lève une exception).
    """
    # 1) Chemin Responses (OpenAI SDK v1.x récent)
    try:
        if hasattr(client, "responses"):
            r = client.responses.create(
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
            try:
                return r.output[0].content[0].text
            except Exception:
                return getattr(r, "output_text", "")
    except Exception as e:
        print("ℹ️ responses() non disponible / erreur:", repr(e))

    # 2) Fallback Chat Completions (vision)
    chat = client.chat.completions.create(
        model=FALLBACK_MODEL,  # "gpt-4o-mini"
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        response_format={"type": "json_object"}
    )
    return chat.choices[0].message.content or ""

# ——— Extraction facture ———
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # 0) Mode MOCK gratuit — active si GARANTIX_MOCK=1
    if os.getenv("GARANTIX_MOCK") == "1":
        return mock_payload()

    try:
        # 1) Lecture et vérifs
        blob = await file.read()
        if not blob:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(blob) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(blob)//1024} KB). Try < {MAX_UPLOAD_BYTES//1024} KB."
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

        text = call_openai_vision_json(data_url, prompt)
        if not text:
            raise HTTPException(status_code=502, detail="Empty response from model")

        # 3) Parsing sécurisé
        try:
            parsed = json.loads(text)
        except Exception:
            print("⚠️ JSON invalide, brut (500 premiers caractères):", text[:500])
            return {"data": {"raw": text}, "warning": "model_did_not_return_valid_json"}

        return {"data": parsed}

    except (httpx.TimeoutException, httpx.ReadTimeout) as e:
        print("⏱️ Timeout OpenAI:", repr(e))
        raise HTTPException(status_code=504, detail="Upstream timeout (try a smaller/clearer file)")
    except HTTPException:
        raise
    except Exception as e:
        print("❌ OpenAI/Server error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Upstream error: {str(e)}")
