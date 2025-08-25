from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, base64, mimetypes, json
from openai import OpenAI
import httpx  # ✅ pour capter les timeouts réseau

app = FastAPI(title="Garantix Extract API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Timeout plus court (Render coupe souvent après ~100s)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=45.0,        # secondes (request total)
    max_retries=1
)

MAX_UPLOAD_BYTES = 6 * 1024 * 1024  # ~6 Mo pour éviter 502 sur gros PDF/images

def to_data_url(content: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        blob = await file.read()
        if not blob:
            raise HTTPException(status_code=400, detail="Empty file")

        # ✅ garde une borne haute (Render/Cloudflare n’aiment pas les très gros bodies)
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
        raise
    except (httpx.TimeoutException, httpx.ReadTimeout) as e:
        print("⏱️ Timeout OpenAI:", repr(e))
        raise HTTPException(status_code=504, detail="Upstream timeout (try a smaller/clearer file)")
    except Exception as e:
        print("❌ OpenAI/Server error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Upstream error: {str(e)}")
