# app.py
import asyncio
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

import uuid
import time

from llm_stylist import (
    TOPS,
    BOTTOMS,
    call_openai_stylist,
    call_openai_stylist_for_bottom,
    match_bottom,
    match_top,
    extract_metadata_from_image,
)

# ------------------------------------------------------
# Data / images: ensure Kaggle dataset exists in /data
# ------------------------------------------------------

DATA_ROOT = Path("/data")
# After extracting the Kaggle dataset, we expect images at /data/images/<id>.jpg
IMAGES_DIR = DATA_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)  # 👈 ensure directory exists
ZIP_PATH = DATA_ROOT / "catalog_images.zip"

HISTORY_DIR = DATA_ROOT / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def ensure_images_dataset() -> None:
    """
    Make sure /data/images exists and contains images.

    If there are already .jpg files there (persistent disk case),
    do nothing. Otherwise, download a zip from DATASET_URL (HuggingFace, etc.)
    and extract it into /data so we get /data/images/<id>.jpg.
    """
    # If images already exist, we're done (disk is warm).
    if IMAGES_DIR.exists():
        jpgs = list(IMAGES_DIR.glob("*.jpg"))
        if jpgs:
            print(f"[dataset] Found {len(jpgs)} images in {IMAGES_DIR}, skipping download.")
            return

    url = os.environ.get("DATASET_URL")
    if not url:
        print(
            "[dataset] No images in /data/images and DATASET_URL is not set. "
            "Service will run, but /static/<id>.jpg will 404 until you upload images."
        )
        return

    print(f"[dataset] No images found, downloading dataset from {url}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with ZIP_PATH.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[dataset] Downloaded zip to {ZIP_PATH}, extracting...")

        # IMPORTANT: this assumes the zip has an `images/` folder at its root,
        # so after extraction you get /data/images/<id>.jpg
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_ROOT)

        print(f"[dataset] Extracted dataset to {DATA_ROOT}")
    except Exception as e:
        print(f"[dataset] WARNING: Failed to download or extract dataset: {e}")
        print(
            "[dataset] Service will still run, but /static/<id>.jpg "
            "will 404 until images exist under /data/images."
        )
    finally:
        if ZIP_PATH.exists():
            try:
                ZIP_PATH.unlink()
                print(f"[dataset] Deleted zip {ZIP_PATH}")
            except OSError:
                pass


# ------------------------------------------------------
# FastAPI app
# ------------------------------------------------------

app = FastAPI(
    title="Virtual Try-On Stylist API",
    description="LLM + Vision + Catalog backend for CIS 5810 final project",
    version="1.0.0",
)

# Run dataset download in the background after the server starts
@app.on_event("startup")
async def startup_populate_images():
    # Fire-and-forget background task; do NOT await
    loop = asyncio.get_event_loop()
    loop.create_task(asyncio.to_thread(ensure_images_dataset))
    print("[startup] Scheduled background dataset check/download.")

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
# /static/<id>.jpg -> /data/images/<id>.jpg
# ------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(IMAGES_DIR)), name="static")
app.mount("/history_static", StaticFiles(directory=str(HISTORY_DIR)), name="history_static")

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

@app.post("/history/upload")
async def upload_history_image(file: UploadFile = File(...)):
    """
    Save a user try-on image into /data/history.

    Returns:
      - id: internal filename
      - image_url: path that frontend can display directly
    """
    # Read file bytes
    contents = await file.read()

    # Determine extension (default to .jpg)
    orig_name = file.filename or ""
    ext = Path(orig_name).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"

    # Unique filename: <timestamp>_<uuid>.ext
    fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}{ext}"
    dest = HISTORY_DIR / fname

    with dest.open("wb") as f:
        f.write(contents)

    return {
        "id": fname,
        "image_url": f"/history_static/{fname}",
    }


@app.get("/history")
def list_history(limit: int = 20):
    """
    List the most recent user try-on images.

    Query params:
      - limit: max number of images to return (default 20)

    Returns:
      [
        { "id": "...", "image_url": "/history_static/...", "created_at": 1234567890.0 },
        ...
      ]
    """
    files = [p for p in HISTORY_DIR.glob("*") if p.is_file()]
    # newest first
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    items = []
    for p in files[:limit]:
        stat = p.stat()
        items.append({
            "id": p.name,
            "image_url": f"/history_static/{p.name}",
            "created_at": stat.st_mtime,
        })

    return items
