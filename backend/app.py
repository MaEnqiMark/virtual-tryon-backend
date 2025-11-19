# app.py

import os
from pathlib import Path
import zipfile

import requests
import pandas as pd
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_stylist import (
    TOPS, BOTTOMS,
    call_openai_stylist, call_openai_stylist_for_bottom,
    match_bottom, match_top,
    extract_metadata_from_image,
)

# ------------------------------------------------------
# Data / images: ensure dataset exists in /data
# ------------------------------------------------------

DATA_ROOT = Path("/data")
IMAGES_DIR = DATA_ROOT / "catalog" / "images"
ZIP_PATH = DATA_ROOT / "catalog_images.zip"


def ensure_images_dataset() -> None:
    """
    Make sure /data/catalog/images exists and contains images.

    If there are already .jpg files there (persistent disk case),
    do nothing. Otherwise, download a zip from DATASET_URL and
    extract it into /data/catalog/images.
    """
    # If images already exist, we're done (disk is warm).
    if IMAGES_DIR.exists():
        jpgs = list(IMAGES_DIR.glob("*.jpg"))
        if jpgs:
            print(f"[dataset] Found {len(jpgs)} images in {IMAGES_DIR}, skipping download.")
            return

    # No images yet: need to download.
    url = os.environ.get("DATASET_URL")
    if not url:
        raise RuntimeError(
            "[dataset] No images in /data/catalog/images and DATASET_URL is not set. "
            "Set DATASET_URL to a zip file URL containing your catalog images."
        )

    print(f"[dataset] No images found, downloading dataset from {url}")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Download zip
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with ZIP_PATH.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"[dataset] Downloaded zip to {ZIP_PATH}, extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(IMAGES_DIR)

    print(f"[dataset] Extracted images to {IMAGES_DIR}")
    # Optional: you can delete the zip to save space
    try:
        ZIP_PATH.unlink()
        print(f"[dataset] Deleted zip {ZIP_PATH}")
    except FileNotFoundError:
        pass


# Ensure dataset before creating the app / mounting static files
ensure_images_dataset()

# ------------------------------------------------------
# FastAPI app
# ------------------------------------------------------

app = FastAPI(
    title="Virtual Try-On Stylist API",
    description="LLM + Vision + Catalog backend for CIS 5810 final project",
    version="1.0.0",
)

# ------------------------------------------------------
# CORS (so your teammate's frontend can call the API)
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# Static files: serve catalog images
# /static/<id>.jpg -> /data/catalog/images/<id>.jpg
# ------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(IMAGES_DIR)), name="static")


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------

def _safe(v):
    """Convert NaN/NA to None so JSON encoding doesn't die."""
    return None if pd.isna(v) else v


def _row_to_item(row, kind: str):
    """Convert a pandas row (top or bottom) to a JSON-friendly dict."""
    return {
        "kind": kind,
        "id": int(row.id),
        "articleType": _safe(row.articleType),
        "baseColour": _safe(row.baseColour),
        "season": _safe(row.season),
        "usage": _safe(row.usage),
        "image_url": f"/static/{int(row.id)}.jpg",
    }


# ------------------------------------------------------
# Pydantic request models
# ------------------------------------------------------

class GenerateLooksFromTopRequest(BaseModel):
    top_id: int
    occasion: Optional[str] = "casual"
    vibe: Optional[str] = "minimal"


class GenerateLooksFromBottomRequest(BaseModel):
    bottom_id: int
    occasion: Optional[str] = "casual"
    vibe: Optional[str] = "minimal"


# ------------------------------------------------------
# Basic endpoints
# ------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/tops")
def list_tops():
    """Return all tops in the catalog with basic metadata."""
    return [_row_to_item(row, "top") for _, row in TOPS.iterrows()]


@app.get("/bottoms")
def list_bottoms():
    """Return all bottoms in the catalog with basic metadata."""
    return [_row_to_item(row, "bottom") for _, row in BOTTOMS.iterrows()]


# ------------------------------------------------------
# Top -> Bottoms
# ------------------------------------------------------

@app.post("/generate_looks_from_top")
def generate_looks_from_top(req: GenerateLooksFromTopRequest):
    """
    Given a TOP id, use the stylist LLM to generate bottom constraints,
    match them to actual bottoms in the catalog, and return outfits.
    """
    top_rows = TOPS[TOPS["id"] == req.top_id]
    if top_rows.empty:
        raise HTTPException(status_code=404, detail=f"Top id {req.top_id} not found")

    top_row = top_rows.iloc[0]

    style_plan = call_openai_stylist(
        top_row,
        num_outfits=3,
        occasion=req.occasion,
        vibe=req.vibe,
    )

    looks = []
    for outfit in style_plan["outfits"]:
        constraint = outfit["bottom_constraint"]
        bottom_row = match_bottom(constraint, BOTTOMS)
        looks.append({
            "name": outfit["name"],
            "constraint": constraint,
            "bottom": _row_to_item(bottom_row, "bottom"),
        })

    return {
        "top": _row_to_item(top_row, "top"),
        "explanation": style_plan.get("explanation", ""),
        "looks": looks,
    }


# ------------------------------------------------------
# Bottom -> Tops
# ------------------------------------------------------

@app.post("/generate_looks_from_bottom")
def generate_looks_from_bottom(req: GenerateLooksFromBottomRequest):
    """
    Given a BOTTOM id, use the stylist LLM to generate top constraints,
    match them to actual tops in the catalog, and return outfits.
    """
    bottom_rows = BOTTOMS[BOTTOMS["id"] == req.bottom_id]
    if bottom_rows.empty:
        raise HTTPException(status_code=404, detail=f"Bottom id {req.bottom_id} not found")

    bottom_row = bottom_rows.iloc[0]

    style_plan = call_openai_stylist_for_bottom(
        bottom_row,
        num_outfits=3,
        occasion=req.occasion,
        vibe=req.vibe,
    )

    looks = []
    for outfit in style_plan["outfits"]:
        constraint = outfit["top_constraint"]
        top_row = match_top(constraint, TOPS)
        looks.append({
            "name": outfit["name"],
            "constraint": constraint,
            "top": _row_to_item(top_row, "top"),
        })

    return {
        "bottom": _row_to_item(bottom_row, "bottom"),
        "explanation": style_plan.get("explanation", ""),
        "looks": looks,
    }


# ------------------------------------------------------
# Upload photo -> auto-detect (top / bottom) -> suggestions
# ------------------------------------------------------

@app.post("/suggest_from_photo")
async def suggest_from_photo(file: UploadFile = File(...)):
    """
    User uploads a photo of a single clothing item (top or bottom).

    We:
      1. Use OpenAI vision to infer metadata (garment_kind, articleType, baseColour,
         season, usage, styleVibe).
      2. Infer occasion and vibe from that metadata.
      3. If top: use stylist to suggest bottoms.
         If bottom: use stylist to suggest tops.
    """
    image_bytes = await file.read()

    # 1) Vision: infer structured metadata
    meta = extract_metadata_from_image(image_bytes)
    garment_kind = meta.get("garment_kind")

    if garment_kind not in ("top", "bottom"):
        raise HTTPException(status_code=400, detail=f"Unclear garment_kind: {garment_kind}")

    # 2) Infer occasion + vibe from metadata
    occasion = meta.get("usage", "Casual")
    vibe = meta.get("styleVibe", "minimal")

    # 3) Build a fake row with the fields the stylist expects
    fake_row = pd.Series({
        "articleType": meta.get("articleType", ""),
        "baseColour": meta.get("baseColour", ""),
        "season": meta.get("season", ""),
        "usage": meta.get("usage", ""),
    })

    looks = []
    explanation = ""

    if garment_kind == "top":
        # TOP -> suggest BOTTOMS
        style_plan = call_openai_stylist(
            fake_row,
            num_outfits=3,
            occasion=occasion,
            vibe=vibe,
        )
        explanation = style_plan.get("explanation", "")

        for outfit in style_plan["outfits"]:
            constraint = outfit["bottom_constraint"]
            bottom_row = match_bottom(constraint, BOTTOMS)
            looks.append({
                "name": outfit["name"],
                "constraint": constraint,
                "bottom": _row_to_item(bottom_row, "bottom"),
            })

        input_item = {
            "kind": "top",
            "articleType": fake_row["articleType"],
            "baseColour": fake_row["baseColour"],
            "season": fake_row["season"],
            "usage": fake_row["usage"],
            "styleVibe": meta.get("styleVibe"),
        }

    else:
        # BOTTOM -> suggest TOPS
        style_plan = call_openai_stylist_for_bottom(
            fake_row,
            num_outfits=3,
            occasion=occasion,
            vibe=vibe,
        )
        explanation = style_plan.get("explanation", "")

        for outfit in style_plan["outfits"]:
            constraint = outfit["top_constraint"]
            top_row = match_top(constraint, TOPS)
            looks.append({
                "name": outfit["name"],
                "constraint": constraint,
                "top": _row_to_item(top_row, "top"),
            })

        input_item = {
            "kind": "bottom",
            "articleType": fake_row["articleType"],
            "baseColour": fake_row["baseColour"],
            "season": fake_row["season"],
            "usage": fake_row["usage"],
            "styleVibe": meta.get("styleVibe"),
        }

    return {
        "input_metadata": meta,
        "inferred_occasion": occasion,
        "inferred_vibe": vibe,
        "normalized_item": input_item,
        "explanation": explanation,
        "looks": looks,
    }
