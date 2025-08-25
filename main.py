from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os, base64, mimetypes, json
from openai import OpenAI
import httpx

app = FastAPI(title="Garantix Extract API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 6 * 1024 * 1024
OPENAI_MODEL = "gpt-4.1-mini"
FALLBACK_MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=45.0,
    max_retries=1
)

@app.get("/")
def hello():
    return {"status": "ok", "message": "Garantix backend en ligne"}

@app.get("/health")
def health():
    return {"ok": True}

# 🆕 Endpoint diagnostic config
@app.get("/config")
def config():
    return {
        "mock_env": os.getenv("GARANTIX_MOCK"),
        "mock_active": os.getenv("GARANTIX_MOCK") == "1"
    }

# -------- Utils --------
def to_data_url(content: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def mock_payload():
    return {
        "data": {
            "seller": {"name": "TEXIPOOL SASU"},
            "purchase_date": "2025-08-20",
            "currency": "EUR",
            "totals": {"total_ttc": 778.64},
            "items": [
                {"product_name": "Enrouleur piscine 3-5 m", "line_total": 190.0,
                 "warranty": {"duration_months": 24}, "confidence": 0.92},
                {"product_name": "Bâche bulles 500µ Noir", "line_total": 530.96,
                 "warranty": {"duration_months": 48}, "confidence": 0.88}
            ],
            "invoice_number": "290144",
            "payment_method": "PayPal",
            "confidence": 0.9
        }
    }

def call_openai_vision_json(data_url: str, prompt: str) -> str:
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
        print("ℹ️ responses() erreur:", repr(e))

    chat = client.chat.completions.create(
        model=FALLBACK_MODEL,
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

# -------- Endpoint extract --------
@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    mock: int = Query(default=0, description="1 = forcer le mode mock gratuit")
):
    # ✅ Mode mock forcé par paramètre ou variable d'env
    if mock == 1 or os.getenv("GARANTIX_MOCK") == "1":
        return mock_payload()

    try:
        blob = await file.read()
        if not blob:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(blob) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="File too large")

        api_key = os.getenv("OPENAI_API_KEY") or ""
        if not api_key.startswith(("sk-", "sk-proj-")):
            raise HTTPException(500, detail="OPENAI_API_KEY missing/invalid")

        data_url = to_data_url(blob, file.filename)

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
            raise HTTPException(502, detail="Empty response from model")

        try:
            parsed = json.loads(text)
        except Exception:
            return {"data": {"raw": text}, "warning": "invalid JSON"}

        return {"data": parsed}

    except (httpx.TimeoutException, httpx.ReadTimeout):
        raise HTTPException(504, detail="OpenAI timeout")
    except HTTPException:
        raise
    except Exception as e:
        print("❌ Server error:", repr(e))
        raise HTTPException(500, detail=str(e))
