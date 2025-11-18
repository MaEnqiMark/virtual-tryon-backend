# llm_stylist.py

import os
import json
import random
import base64
import pandas as pd
from openai import OpenAI

MODEL_VISION = "gpt-4o-mini"
MODEL_STYLIST = "gpt-4o-mini"     # You can upgrade to gpt-4.1 if you want

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# 1. Load catalog
# ============================================================

CSV_PATH = "catalog/styles_subset.csv"
IMAGES_DIR = "catalog/images"

df = pd.read_csv(CSV_PATH)
df["image_path"] = df["id"].apply(lambda x: f"{IMAGES_DIR}/{int(x)}.jpg")

TOPS = df[df["subCategory"].str.contains("Topwear", case=False, na=False)].copy()
BOTTOMS = df[df["subCategory"].str.contains("Bottomwear", case=False, na=False)].copy()

print("[llm_stylist] Loaded catalog")
print("[llm_stylist]   TOPS:", len(TOPS))
print("[llm_stylist]   BOTTOMS:", len(BOTTOMS))


# ============================================================
# 2. FIXED VOCABULARY (YOUR DATASET VALUES)
# ============================================================

VALID_ARTICLE_TYPES_TOP = [
    "Tshirts", "Shirts", "Sweatshirts", "Jackets", "Hoodies", "Kurtas",
    "Blouses", "Top", "Coats", "Sweaters"
]

VALID_ARTICLE_TYPES_BOTTOM = [
    "Jeans", "Trousers", "Pants", "Shorts", "Skirts", "Track Pants", "Joggers"
]

VALID_COLOURS = [
    "Black", "White", "Grey", "Blue", "Navy Blue", "Red", "Pink",
    "Yellow", "Beige", "Green", "Brown"
]

VALID_SEASONS = ["Summer", "Winter", "Fall", "Spring"]
VALID_USAGE = ["Casual", "Formal", "Sports", "Party"]
VALID_VIBES = ["minimal", "streetwear", "sporty", "preppy", "casual", "classy"]


def nearest_choice(value: str, choices: list[str]) -> str:
    """Return the closest allowed value (simple containment match)."""
    if not value:
        return random.choice(choices)

    val = value.lower()
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
            # part may be an object or a dict
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
    response_format={"type": "json_object"},  # <--- force JSON
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
        # Debug help: print raw so you can see what went wrong
        print("=== RAW VISION RESPONSE ===")
        print(repr(raw))
        raise

    # normalize to fixed sets...
    if data["garment_kind"] == "top":
        data["articleType"] = nearest_choice(data["articleType"], VALID_ARTICLE_TYPES_TOP)
    else:
        data["articleType"] = nearest_choice(data["articleType"], VALID_ARTICLE_TYPES_BOTTOM)

    data["baseColour"] = nearest_choice(data["baseColour"], VALID_COLOURS)
    data["season"] = nearest_choice(data["season"], VALID_SEASONS)
    data["usage"] = nearest_choice(data["usage"], VALID_USAGE)
    data["styleVibe"] = nearest_choice(data["styleVibe"], VALID_VIBES)

    return data


# ============================================================
# 4. LLM STYLIST (TOP → BOTTOMS)
# ============================================================

def call_openai_stylist(top_row: pd.Series, num_outfits=3, occasion=None, vibe=None):
    system_prompt = f"""
You are a fashion stylist. You receive a TOP and must suggest complementary BOTTOMS.

Return ONLY JSON:

{{
  "explanation": "short explanation",
  "outfits": [
    {{
      "name": "short name",
      "bottom_constraint": {{
        "articleType": from {VALID_ARTICLE_TYPES_BOTTOM},
        "baseColour": from {VALID_COLOURS},
        "season": from {VALID_SEASONS},
        "usage": from {VALID_USAGE}
      }}
    }},
    ...
  ]
}}

Generate EXACTLY {num_outfits} outfits.
"""

    user_prompt = f"""
Top metadata:
- articleType: {top_row['articleType']}
- baseColour: {top_row['baseColour']}
- season: {top_row['season']}
- usage: {top_row['usage']}
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
    return json.loads(raw)


# ============================================================
# 5. LLM STYLIST (BOTTOM → TOPS)
# ============================================================

def call_openai_stylist_for_bottom(bottom_row: pd.Series, num_outfits=3, occasion=None, vibe=None):
    system_prompt = f"""
You are a fashion stylist. You receive a BOTTOM and must suggest complementary TOPS.

Return ONLY JSON:

{{
  "explanation": "short explanation",
  "outfits": [
    {{
      "name": "short name",
      "top_constraint": {{
        "articleType": from {VALID_ARTICLE_TYPES_TOP},
        "baseColour": from {VALID_COLOURS},
        "season": from {VALID_SEASONS},
        "usage": from {VALID_USAGE}
      }}
    }},
    ...
  ]
}}

Generate EXACTLY {num_outfits} outfits.
"""

    user_prompt = f"""
Bottom metadata:
- articleType: {bottom_row['articleType']}
- baseColour: {bottom_row['baseColour']}
- season: {bottom_row['season']}
- usage: {bottom_row['usage']}
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
    return json.loads(raw)


# ============================================================
# 6. Constraint → Actual Catalog Item Matching
# ============================================================

def _match(df: pd.DataFrame, constraint: dict):
    c = df.copy()

    for key in ["articleType", "baseColour", "season", "usage"]:
        if key in constraint and constraint[key]:
            mask = c[key].fillna("").str.contains(constraint[key], case=False)
            filtered = c[mask]
            if not filtered.empty:
                c = filtered

    if c.empty:
        return df.sample(n=1).iloc[0]
    return c.sample(n=1).iloc[0]


def match_bottom(constraint: dict, bottoms=BOTTOMS):
    return _match(bottoms, constraint)


def match_top(constraint: dict, tops=TOPS):
    return _match(tops, constraint)
