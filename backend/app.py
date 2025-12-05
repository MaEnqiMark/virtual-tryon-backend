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
    JACKETS,
    SHOES,
    JACKETS_CSV, SHOES_CSV,
    call_openai_stylist,
    call_openai_stylist_for_bottom,
    match_bottom,
    match_top,
    match_jacket,
    match_shoes,
    extract_metadata_from_image,
)


# =======================================================
# Filesystem Setup
# =======================================================

DATA_ROOT = Path("/data")
IMAGES_DIR = DATA_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DATA_ROOT / "catalog_images.zip"

HISTORY_DIR = DATA_ROOT / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

JACKETS_CSV = DATA_ROOT / "jackets.csv"
SHOES_CSV   = DATA_ROOT / "shoes.csv"
JACKETS_SHOES_ZIP = DATA_ROOT / "jackets_shoes.zip"


def ensure_jackets_shoes_dataset() -> None:

    # ---- Case 1: Already downloaded on disk -> skip download & reload into memory ----
    if JACKETS_CSV.exists() and SHOES_CSV.exists():
        print("[jackets/shoes] Found jackets + shoes CSVs on disk.")
        try:
            import llm_stylist
            llm_stylist.JACKETS = pd.read_csv(JACKETS_CSV)
            llm_stylist.SHOES   = pd.read_csv(SHOES_CSV)
            print("[jackets/shoes] Reloaded JACKETS + SHOES into memory")
        except Exception as e:
            print("[jackets/shoes] Reload failed:", e)
        return

    # ---- Case 2: CSV missing -> must download ----
    url = os.environ.get("JACKETS_SHOES_URL")
    if not url:
        print("[jackets/shoes] JACKETS_SHOES_URL not set; skipping.")
        return

    print(f"[jackets/shoes] Downloading from {url}")
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with JACKETS_SHOES_ZIP.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        with zipfile.ZipFile(JACKETS_SHOES_ZIP, "r") as zip_ref:
            zip_ref.extractall(DATA_ROOT)

        print("[jackets/shoes] Extracted successfully.")

        # Reload into memory after extraction
        try:
            import llm_stylist
            llm_stylist.JACKETS = pd.read_csv(JACKETS_CSV)
            llm_stylist.SHOES   = pd.read_csv(SHOES_CSV)
            print("[jackets/shoes] Reloaded JACKETS + SHOES into memory after download")
        except Exception as e:
            print("[jackets/shoes] Reload failed after download:", e)

    except Exception as e:
        print(f"[jackets/shoes] Error: {e}")

    finally:
        if JACKETS_SHOES_ZIP.exists():
            JACKETS_SHOES_ZIP.unlink()


def ensure_images_dataset() -> None:
    if IMAGES_DIR.exists():
        jpgs = list(IMAGES_DIR.glob("*.jpg"))
        if jpgs:
            print(f"[dataset] Found {len(jpgs)} images, skipping download.")
            return

    url = os.environ.get("DATASET_URL")
    if not url:
        print("[dataset] DATASET_URL not set — catalog images missing.")
        return

    print(f"[dataset] Downloading from {url}")
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with ZIP_PATH.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_ROOT)

        print("[dataset] Extracted successfully.")
    except Exception as e:
        print(f"[dataset] Error downloading: {e}")
    finally:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()


# =======================================================
# FastAPI
# =======================================================

app = FastAPI(
    title="Virtual Try-On Stylist API",
    version="2.0.0",
)

@app.on_event("startup")
async def startup_tasks():
    loop = asyncio.get_event_loop()
    loop.create_task(asyncio.to_thread(ensure_images_dataset))
    loop.create_task(asyncio.to_thread(ensure_jackets_shoes_dataset))
    print("[startup] checks scheduled.")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static routes
app.mount("/static", StaticFiles(directory=str(IMAGES_DIR)), name="static")
app.mount("/history_static", StaticFiles(directory=str(HISTORY_DIR)), name="history_static")


# =======================================================
# Helpers
# =======================================================

def _safe(v):
    return None if pd.isna(v) else v

def _row_to_item(row, kind: str):
    return {
        "kind": kind,
        "id": int(row.id),
        "articleType": _safe(row.articleType),
        "baseColour": _safe(row.baseColour),
        "season": _safe(row.season),
        "usage": _safe(row.usage),
        "image_url": f"/static/{int(row.id)}.jpg",
    }

def _append_jackets_shoes(looks_list: list, constraint: dict):
    # 2 jackets
    for i in range(2):
        try:
            j = match_jacket(constraint) or (JACKETS.sample(n=1).iloc[0] if not JACKETS.empty else None)
        except:
            j = None
        if j is not None:
            looks_list.append({
                "name": f"Jacket {i+1}",
                "constraint": constraint,
                "bottom": None,
                "top": None,
                "item": _row_to_item(j, "jacket"),
            })

    # 2 shoes
    for i in range(2):
        try:
            s = match_shoes(constraint) or (SHOES.sample(n=1).iloc[0] if not SHOES.empty else None)
        except:
            s = None
        if s is not None:
            looks_list.append({
                "name": f"Shoe {i+1}",
                "constraint": constraint,
                "bottom": None,
                "top": None,
                "item": _row_to_item(s, "shoes"),
            })



# =======================================================
# Schemas
# =======================================================

class GenerateLooksFromTopRequest(BaseModel):
    top_id: int
    occasion: Optional[str] = "casual"
    vibe: Optional[str] = "minimal"


class GenerateLooksFromBottomRequest(BaseModel):
    bottom_id: int
    occasion: Optional[str] = "casual"
    vibe: Optional[str] = "minimal"


# =======================================================
# Endpoints
# =======================================================

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/tops")
def list_tops():
    return [_row_to_item(row, "top") for _, row in TOPS.iterrows()]


@app.get("/bottoms")
def list_bottoms():
    return [_row_to_item(row, "bottom") for _, row in BOTTOMS.iterrows()]


@app.post("/generate_looks_from_top")
def generate_looks_from_top(req: GenerateLooksFromTopRequest):
    rows = TOPS[TOPS["id"] == req.top_id]
    if rows.empty:
        raise HTTPException(404, f"Top id {req.top_id} not found")
    top_row = rows.iloc[0]

    style_plan = call_openai_stylist(top_row, occasion=req.occasion, vibe=req.vibe)

    looks = []
    for outfit in style_plan["outfits"]:
        constraint = outfit["bottom_constraint"]
        matched = match_bottom(constraint)
        looks.append({
            "name": outfit["name"],
            "constraint": constraint,
            "bottom": _row_to_item(matched, "bottom"),
        })

    # Add 2 jackets + 2 shoes, rendered as extra "bottom" options
    constraint_input = {
        "articleType": top_row.articleType,
        "baseColour": top_row.baseColour,
        "season": top_row.season,
        "usage": top_row.usage,
    }
    _append_jackets_shoes(looks, constraint_input)

    return {
        "top": _row_to_item(top_row, "top"),
        "explanation": style_plan.get("explanation", ""),
        "looks": looks,
    }


@app.post("/generate_looks_from_bottom")
def generate_looks_from_bottom(req: GenerateLooksFromBottomRequest):
    rows = BOTTOMS[BOTTOMS["id"] == req.bottom_id]
    if rows.empty:
        raise HTTPException(404, f"Bottom id {req.bottom_id} not found")
    bottom_row = rows.iloc[0]

    style_plan = call_openai_stylist_for_bottom(
        bottom_row,
        occasion=req.occasion,
        vibe=req.vibe,
    )

    looks = []
    for outfit in style_plan["outfits"]:
        constraint = outfit["top_constraint"]
        matched = match_top(constraint)
        looks.append({
            "name": outfit["name"],
            "constraint": constraint,
            "top": _row_to_item(matched, "top"),
        })

    # Add 2 jackets + 2 shoes, rendered as extra "top" options
    constraint_input = {
        "articleType": bottom_row.articleType,
        "baseColour": bottom_row.baseColour,
        "season": bottom_row.season,
        "usage": bottom_row.usage,
    }
    _append_jackets_shoes(looks, constraint_input)

    return {
        "bottom": _row_to_item(bottom_row, "bottom"),
        "explanation": style_plan.get("explanation", ""),
        "looks": looks,
    }


@app.post("/suggest_from_photo")
async def suggest_from_photo(file: UploadFile = File(...)):
    image_bytes = await file.read()

    meta = extract_metadata_from_image(image_bytes)
    garment_kind = meta.get("garment_kind")

    fake_row = pd.Series({
        "articleType": meta.get("articleType", ""),
        "baseColour": meta.get("baseColour", ""),
        "season": meta.get("season", ""),
        "usage": meta.get("usage", ""),
    })

    looks = []
    explanation = ""

    if garment_kind == "top":
        style_plan = call_openai_stylist(fake_row)
        explanation = style_plan.get("explanation", "")

        for outfit in style_plan["outfits"]:
            constraint = outfit["bottom_constraint"]
            matched = match_bottom(constraint)
            looks.append({
                "name": outfit["name"],
                "constraint": constraint,
                "bottom": _row_to_item(matched, "bottom"),
            })

        # add jackets + shoes as extra "bottom" options
        _append_jackets_shoes(looks, meta)

    elif garment_kind == "bottom":
        style_plan = call_openai_stylist_for_bottom(fake_row)
        explanation = style_plan.get("explanation", "")

        for outfit in style_plan["outfits"]:
            constraint = outfit["top_constraint"]
            matched = match_top(constraint)
            looks.append({
                "name": outfit["name"],
                "constraint": constraint,
                "top": _row_to_item(matched, "top"),
            })

        # add jackets + shoes as extra "top" options
        _append_jackets_shoes(looks, meta)

    else:
        raise HTTPException(400, f"Unclear garment_kind: {garment_kind}")

    return {
        "input_metadata": meta,
        "inferred_occasion": meta.get("usage", "Casual"),
        "inferred_vibe": meta.get("styleVibe", "minimal"),
        "normalized_item": {
            "articleType": fake_row["articleType"],
            "baseColour": fake_row["baseColour"],
            "season": fake_row["season"],
            "usage": fake_row["usage"],
            "styleVibe": meta.get("styleVibe"),
            "kind": garment_kind,
        },
        "explanation": explanation,
        "looks": looks,
    }


# =======================================================
# History
# =======================================================

@app.post("/history/upload")
async def upload_history_image(file: UploadFile = File(...)):
    contents = await file.read()

    ext = Path(file.filename or "").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"

    fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}{ext}"
    dest = HISTORY_DIR / fname
    with dest.open("wb") as f:
        f.write(contents)

    return {"id": fname, "image_url": f"/history_static/{fname}"}


@app.get("/history")
def list_history(limit: int = 20):
    files = sorted(
        [p for p in HISTORY_DIR.glob("*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    return [
        {
            "id": p.name,
            "image_url": f"/history_static/{p.name}",
            "created_at": p.stat().st_mtime,
        }
        for p in files
    ]
