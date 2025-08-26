import base64
import json
from io import BytesIO
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI

from PIL import Image, ImageOps
from pdf2image import convert_from_bytes

# (Optionnel) HEIC -> Pillow
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

app = FastAPI(title="Garantix Extractor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

client = OpenAI()  # nécessite OPENAI_API_KEY en variable d'env

ALLOWED_IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"
}

def pil_to_data_url(img: Image.Image) -> str:
    """Convertit une image PIL en data URL PNG base64 (robuste)."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def normalize_image(content: bytes) -> Image.Image:
    """Ouvre l'image, corrige l'orientation EXIF, convertit en RGB, limite la taille."""
    img = Image.open(BytesIO(content))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    # Limite la taille pour coût/latence
    max_side = 2000
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / float(max(w, h))
        img = img.resize((int(w * ratio), int(h * ratio)))
    return img

def pdf_to_images(pdf_bytes: bytes, max_pages: int = 3, dpi: int = 220) -> List[Image.Image]:
    """Convertit un PDF en images (PNG) et limite à max_pages."""
    pages = convert_from_bytes(pdf_bytes, fmt="png", dpi=dpi)
    return pages[:max_pages]

def build_prompt_schema() -> str:
    schema = {
        "vendor": "string",
        "invoice_number": "string|null",
        "date": "YYYY-MM-DD",
        "currency": "string",
        "subtotal": "number|null",
        "tax": "number|null",
        "total": "number",
        "vat_id": "string|null",
        "line_items": [
            {"description": "string", "qty": "number|null", "unit_price": "number|null", "amount": "number|null"}
        ],
    }
    return json.dumps(schema, ensure_ascii=False)

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Garantix Extractor OK"

@app.get("/healthz", response_class=JSONResponse)
def health():
    return {"ok": True}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Accepte: PDF et images (png/jpg/webp/heic).
    Retour: JSON structuré des infos de facture.
    """
    ct = (file.content_type or "").lower()
    name = (file.filename or "").lower()

    # PDF -> to images
    if ct == "application/pdf" or name.endswith(".pdf"):
        pdf_bytes = await file.read()
        try:
            images = pdf_to_images(pdf_bytes, max_pages=3, dpi=220)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF invalide ou non lisible: {e}")
        contents = [{"type": "input_image", "image_url": pil_to_data_url(im)} for im in images]

    # Images -> normalize
    elif ct in ALLOWED_IMAGE_MIMES or name.endswith((".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")):
        raw = await file.read()
        try:
            img = normalize_image(raw)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image invalide: {e}")
        contents = [{"type": "input_image", "image_url": pil_to_data_url(img)}]

    else:
        raise HTTPException(
            status_code=415,
            detail="Type non supporté. Envoyez un PDF ou une image (png/jpg/webp/heic)."
        )

    system_instruction = (
        "Tu es un extracteur de données de factures. "
        "Analyse l'image/PDF et renvoie UNIQUEMENT un JSON valide respectant ce schéma. "
        "Ne rajoute pas de texte autour du JSON."
    )
    user_instruction = (
        "Extrait les informations de facture (vendeur, date, total, TVA, lignes) en JSON. "
        f"Respecte ce schéma: {build_prompt_schema()} "
        "Utilise un point comme séparateur décimal. Déduis la devise si affichée (€, EUR, $, etc.)."
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[{
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}]
            },{
                "role": "user",
                "content": [{"type": "input_text", "text": user_instruction}] + contents
            }],
            temperature=0
        )

        output_text = getattr(resp, "output_text", None)
        if not output_text:
            try:
                output_text = resp.output[0].content[0].text  # fallback
            except Exception:
                raise HTTPException(status_code=502, detail="Réponse modèle inattendue.")

        # Valide JSON
        try:
            data = json.loads(output_text)
        except json.JSONDecodeError:
            start = output_text.find("{")
            end = output_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(output_text[start:end+1])
            else:
                raise HTTPException(status_code=502, detail="Le modèle n'a pas renvoyé un JSON propre.")

        return {"ok": True, "data": data}

    except HTTPException:
        raise
    except Exception as e:
        # Propager clairement l’erreur amont au client (400 plutôt que 500 opaque)
        raise HTTPException(status_code=400, detail=str(e))
