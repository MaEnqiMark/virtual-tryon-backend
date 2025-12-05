# llm_stylist.py

import os
import json
import random
import base64
import pandas as pd
from openai import OpenAI
from pathlib import Path

# ============================================================
# Models
# ============================================================

MODEL_VISION = "gpt-4o-mini"
MODEL_STYLIST = "gpt-4.1"   # upgraded stylist model

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# 1. Load catalog
# ============================================================

DATA_ROOT = Path("/data")
CSV_PATH = DATA_ROOT / "styles_subset.csv"
IMAGES_DIR = DATA_ROOT / "images"

df = pd.read_csv(CSV_PATH)
df["image_path"] = df["id"].apply(lambda x: f"{IMAGES_DIR}/{int(x)}.jpg")

TOPS = df[df["subCategory"].str.contains("Topwear", case=False, na=False)].copy()
BOTTOMS = df[df["subCategory"].str.contains("Bottomwear", case=False, na=False)].copy()

print("[llm_stylist] Loaded main catalog")
print("[llm_stylist]   TOPS:", len(TOPS))
print("[llm_stylist]   BOTTOMS:", len(BOTTOMS))

# Jackets + Shoes support (for accessories driven by input item)
JACKETS_CSV = DATA_ROOT / "jackets.csv"
SHOES_CSV   = DATA_ROOT / "shoes.csv"

try:
    JACKETS = pd.read_csv(JACKETS_CSV)
    SHOES   = pd.read_csv(SHOES_CSV)
    print("[llm_stylist]   JACKETS:", len(JACKETS))
    print("[llm_stylist]   SHOES:", len(SHOES))
except FileNotFoundError:
    JACKETS = pd.DataFrame()
    SHOES   = pd.DataFrame()
    print("[llm_stylist] WARNING: jackets.csv / shoes.csv not found. Accessories will be empty.")


# ============================================================
# 2. FIXED VOCABULARY (YOUR DATASET VALUES)
# ============================================================

VALID_ARTICLE_TYPES_TOP = [
    "Tshirts", "Shirts", "Sweatshirts", "Jackets", "Hoodies", "Kurtas",
    "Blouses", "Top", "Coats", "Sweaters",
]

VALID_ARTICLE_TYPES_BOTTOM = [
    "Jeans", "Trousers", "Pants", "Shorts", "Skirts", "Track Pants", "Joggers",
]

VALID_COLOURS = [
    "Black", "White", "Grey", "Blue", "Navy Blue", "Red", "Pink",
    "Yellow", "Beige", "Green", "Brown",
]

VALID_SEASONS = ["Summer", "Winter", "Fall", "Spring"]
VALID_USAGE = ["Casual", "Formal", "Sports", "Party"]
VALID_VIBES = ["minimal", "streetwear", "sporty", "preppy", "casual", "classy"]


def nearest_choice(value, choices):
    """Return the closest allowed value (simple containment match) or random."""
    if not value:
        return random.choice(choices)

    val = str(value).lower()
    for c in choices:
        if c.lower() in val:
            return c
    return random.choice(choices)


def _extract_text_from_chat(resp) -> str:
    """
    Normalize OpenAI chat.completions response content into a plain string.
    Handles both string and list-of-parts formats.
    """
    content = resp.choices[0].message.content

    # Newer SDK often returns a list of content parts
    if isinstance(content, list):
        texts = []
        for part in content:
            t = getattr(part, "text", None)
            if t is None and isinstance(part, dict):
                t = part.get("text")
            if t:
                texts.append(t)
        return "".join(texts)

    if isinstance(content, str):
        return content

    raise ValueError(f"Unexpected message.content type: {type(content)}")


# ============================================================
# 3. IMAGE → STRUCTURED CLOTHING METADATA (VISION)
# ============================================================

def extract_metadata_from_image(image_bytes: bytes) -> dict:
    """
    Use OpenAI vision to analyze the image and return normalized metadata.
    Output:
    {
      "garment_kind": "top" | "bottom",
      "articleType": "...",
      "baseColour": "...",
      "season": "...",
      "usage": "...",
      "styleVibe": "..."
    }
    """

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{b64}"

    system_prompt = f"""
You are a fashion vision model. You analyze ONE clothing item in an image.

Return ONLY JSON with the following keys:

{{
  "garment_kind": "top" or "bottom",
  "articleType": string (must be from these: {VALID_ARTICLE_TYPES_TOP + VALID_ARTICLE_TYPES_BOTTOM}),
  "baseColour": string (from: {VALID_COLOURS}),
  "season": one of {VALID_SEASONS},
  "usage": one of {VALID_USAGE},
  "styleVibe": one of {VALID_VIBES}
}}

If something is uncertain, guess. Always choose from the fixed sets.
"""

    user_prompt = "Analyze the item in the image and fill all JSON fields."

    resp = client.chat.completions.create(
        model=MODEL_VISION,
        response_format={"type": "json_object"},  # force JSON
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            },
        ],
        temperature=0.1,
    )

    raw = _extract_text_from_chat(resp).strip()
    if not raw:
        raise RuntimeError("Vision model returned empty content")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("=== RAW VISION RESPONSE ===")
        print(repr(raw))
        raise

    # normalize to fixed sets...
    if data["garment_kind"] == "top":
        data["articleType"] = nearest_choice(data["articleType"], VALID_ARTICLE_TYPES_TOP)
    else:
        data["articleType"] = nearest_choice(data["articleType"], VALID_ARTICLE_TYPES_BOTTOM)

    data["baseColour"] = nearest_choice(data.get("baseColour"), VALID_COLOURS)
    data["season"] = nearest_choice(data.get("season"), VALID_SEASONS)
    data["usage"] = nearest_choice(data.get("usage"), VALID_USAGE)
    data["styleVibe"] = nearest_choice(data.get("styleVibe"), VALID_VIBES)

    return data


# ============================================================
# 4. LLM STYLIST (TOP → BOTTOMS)
# ============================================================

def call_openai_stylist(top_row: pd.Series, num_outfits: int = 3,
                        occasion: str | None = None, vibe: str | None = None) -> dict:
    """
    Given a TOP row, ask the LLM to produce:
      - explanation: string (we'll make it richer, but it's still just a string)
      - outfits: list of objects with "name" and "bottom_constraint"

    JSON SHAPE (unchanged from your original):

    {
      "explanation": "string",
      "outfits": [
        {
          "name": "short name",
          "bottom_constraint": {
            "articleType": from VALID_ARTICLE_TYPES_BOTTOM,
            "baseColour": from VALID_COLOURS,
            "season": from VALID_SEASONS,
            "usage": from VALID_USAGE
          }
        },
        ...
      ]
    }
    """

    system_prompt = f"""
You are a professional fashion stylist.
You receive metadata for a TOP and must suggest complementary BOTTOMS.

Return ONLY JSON in this format:

{{
  "explanation": "A single paragraph of about 5 sentences, using occasional professional
                  fashion concepts such as 'silhouette balance', 'monochromatic palette',
                  or 'textural contrast', and briefly explaining each term in parentheses
                  the first time it appears.",
  "outfits": [
    {{
      "name": "short descriptive name for this outfit",
      "bottom_constraint": {{
        "articleType": one of {VALID_ARTICLE_TYPES_BOTTOM},
        "baseColour": one of {VALID_COLOURS},
        "season": one of {VALID_SEASONS},
        "usage": one of {VALID_USAGE}
      }}
    }},
    ...
  ]
}}

Rules:
- Generate EXACTLY {num_outfits} outfits.
- Use ONLY the allowed categorical values.
"""

    user_prompt = f"""
Top metadata:
- articleType: {top_row.get('articleType', '')}
- baseColour: {top_row.get('baseColour', '')}
- season: {top_row.get('season', '')}
- usage: {top_row.get('usage', '')}
Occasion: {occasion}
Vibe: {vibe}
"""

    resp = client.chat.completions.create(
        model=MODEL_STYLIST,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    raw = _extract_text_from_chat(resp).strip()
    data = json.loads(raw)

    # safety defaults
    data.setdefault("explanation", "")
    data.setdefault("outfits", [])

    return data


# ============================================================
# 5. LLM STYLIST (BOTTOM → TOPS)
# ============================================================

def call_openai_stylist_for_bottom(bottom_row: pd.Series, num_outfits: int = 3,
                                   occasion: str | None = None, vibe: str | None = None) -> dict:
    """
    Given a BOTTOM row, ask the LLM to produce:
      - explanation: string
      - outfits: list of objects with "name" and "top_constraint"

    JSON SHAPE (unchanged from your original):

    {
      "explanation": "string",
      "outfits": [
        {
          "name": "short name",
          "top_constraint": {
            "articleType": from VALID_ARTICLE_TYPES_TOP,
            "baseColour": from VALID_COLOURS,
            "season": from VALID_SEASONS,
            "usage": from VALID_USAGE
          }
        },
        ...
      ]
    }
    """

    system_prompt = f"""
You are a professional fashion stylist.
You receive metadata for a BOTTOM and must suggest complementary TOPS.

Return ONLY JSON in this format:

{{
  "explanation": "A single paragraph of about 5 sentences, with a few professional
                  fashion concepts (and short parenthetical explanations).",
  "outfits": [
    {{
      "name": "short descriptive name for this outfit",
      "top_constraint": {{
        "articleType": one of {VALID_ARTICLE_TYPES_TOP},
        "baseColour": one of {VALID_COLOURS},
        "season": one of {VALID_SEASONS},
        "usage": one of {VALID_USAGE}
      }}
    }},
    ...
  ]
}}

Rules:
- Generate EXACTLY {num_outfits} outfits.
- Use ONLY the allowed categorical values.
"""

    user_prompt = f"""
Bottom metadata:
- articleType: {bottom_row.get('articleType', '')}
- baseColour: {bottom_row.get('baseColour', '')}
- season: {bottom_row.get('season', '')}
- usage: {bottom_row.get('usage', '')}
Occasion: {occasion}
Vibe: {vibe}
"""

    resp = client.chat.completions.create(
        model=MODEL_STYLIST,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    raw = _extract_text_from_chat(resp).strip()
    data = json.loads(raw)

    data.setdefault("explanation", "")
    data.setdefault("outfits", [])

    return data


# ============================================================
# 6. Constraint → Actual Catalog Item Matching
# ============================================================

def _match(df: pd.DataFrame, constraint: dict):
    """
    Filter df by any of [articleType, baseColour, season, usage] present in constraint.
    If filtered result is empty, fall back to the full df.
    Return ONE random row (Series).
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe passed to _match")

    c = df.copy()

    for key in ["articleType", "baseColour", "season", "usage"]:
        if key in constraint and constraint[key]:
            mask = c[key].fillna("").str.contains(str(constraint[key]), case=False)
            filtered = c[mask]
            if not filtered.empty:
                c = filtered

    if c.empty:
        c = df

    return c.sample(n=1).iloc[0]


def match_bottom(constraint: dict, bottoms: pd.DataFrame = BOTTOMS):
    return _match(bottoms, constraint)


def match_top(constraint: dict, tops: pd.DataFrame = TOPS):
    return _match(tops, constraint)


def match_jacket(constraint: dict, jackets: pd.DataFrame = JACKETS):
    """
    Match a single jacket to a constraint (based on input item).
    If JACKETS is empty, return None so app.py can handle it gracefully.
    """
    if jackets is None or jackets.empty:
        return None
    return _match(jackets, constraint)


def match_shoes(constraint: dict, shoes: pd.DataFrame = SHOES):
    """
    Match a single shoes item to a constraint (based on input item).
    If SHOES is empty, return None.
    """
    if shoes is None or shoes.empty:
        return None
    return _match(shoes, constraint)
