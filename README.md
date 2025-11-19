🎽 Virtual Try-On Backend (CIS 5810)

A FastAPI backend that powers our virtual try-on / outfit recommendation system.
It combines:

🧠 LLM-based style reasoning

👗 Photo-based garment classification

🗂️ A curated dataset of tops & bottoms

🖼️ Static serving of clothing images

🚀 Cloud deployment on Render

Live backend URL:
👉 https://virtual-tryon-backend-974u.onrender.com

(You can call this from any frontend — CORS is open.)

📦 Features
✔️ Clothing Catalog API

Browse all tops and bottoms

Metadata includes color, article type, season, usage

Each item includes an image URL for easy display

✔️ Outfit Recommendations

Select a top → get bottom suggestions

Select a bottom → get top suggestions

Powered by OpenAI’s LLM with normalized fashion metadata

✔️ Upload a Photo → Detect Item → Generate Outfit

Upload any clothing image

Vision model extracts garment kind + attributes

System auto-generates 3 looks

Fully integrated into the same recommendation engine

✔️ Static Image Serving

Images are stored on Render’s persistent mounted disk at /data/images,
but served to clients through the same clean, static endpoint:

/static/<id>.jpg


So yes, they behave exactly like normal static assets
even though the files are physically stored in a /data volume.

🛠️ API Endpoints
GET /health

Check if the backend is alive.

Response:

{"status": "ok"}

GET /tops

Returns all topwear items from the catalog.

Example:

[
  {
    "kind": "top",
    "id": 4865,
    "articleType": "Tshirts",
    "baseColour": "White",
    "season": "Summer",
    "usage": "Sports",
    "image_url": "/static/4865.jpg"
  }
]

GET /bottoms

Same format as /tops, but for bottoms.

POST /generate_looks_from_top

Generate outfit recommendations based on a selected top.

Request:
{
  "top_id": 4865,
  "occasion": "casual",
  "vibe": "minimal"
}

Response:

List of bottoms + constraints + explanation.

POST /generate_looks_from_bottom

Generate outfit recommendations based on a selected bottom.

Request:
{
  "bottom_id": 7579,
  "occasion": "party",
  "vibe": "streetwear"
}

POST /suggest_from_photo

Upload a clothing photo to detect its type and generate outfits.

Example using cURL:
curl -X POST \
  -F "file=@your_image.jpg" \
  https://virtual-tryon-backend-974u.onrender.com/suggest_from_photo

Response:

Includes:

detected garment attributes

chosen vibe + occasion

3 matched outfits

real catalog items with /static/<id>.jpg URLs

🖼️ Images & Static Files

Even though the images are stored in the mounted persistent volume:

/data/images/


They are served to clients via FastAPI’s static mount:

/static/<id>.jpg


This keeps frontend code simple and consistent.

Example:

<img src="https://virtual-tryon-backend-974u.onrender.com/static/10025.jpg" />

🧾 NEW: User History API (upload + list)

This allows the frontend to save what users tried on before.

POST /history/upload

Upload any user try-on image.

Form field: file (multipart)

Response:

{
  "id": "1700439123123_abc123.jpg",
  "image_url": "/history_static/1700439123123_abc123.jpg"
}


Full URL:

https://virtual-tryon-backend-974u.onrender.com/history_static/<filename>

GET /history?limit=20

Retrieve most recent uploaded images.

Example response:

[
  {
    "id": "1700439123123_abc123.jpg",
    "image_url": "/history_static/1700439123123_abc123.jpg",
    "created_at": 1700439123.12
  }
]

This is perfect for a “What others have tried before” feed.


☁️ Deployment (Render)
Data persistence via disk

A Render Persistent Disk is mounted at:

/data


This allows storing the 5GB fashion dataset without putting it into Git.
The backend automatically:

creates /data/images on startup

checks if images exist

downloads the dataset (Hugging Face / Kaggle / direct link) only if necessary

extracts all .jpg files

serves them statically

Non-blocking startup

Dataset download happens in a background thread so the app can start instantly:

@app.on_event("startup")
async def startup_populate_images():
    loop.create_task(asyncio.to_thread(ensure_images_dataset))

🧰 Technology Stack

FastAPI

Uvicorn

OpenAI (LLM + Vision)

Pandas

Starlette StaticFiles

Render Web Service + Persistent Disk

🧪 Testing Locally

Install dependencies:

pip install -r requirements.txt


Run:

uvicorn app:app --reload


Open:

http://localhost:8000/health

🧑‍🤝‍🧑 For Frontend Developers

You can hit the backend directly from JS/TS:

const res = await fetch(
  "https://virtual-tryon-backend-974u.onrender.com/generate_looks_from_top",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      top_id: 4865,
      occasion: "casual",
      vibe: "minimal"
    })
  }
);

const data = await res.json();


Images load simply via:

<img src="https://virtual-tryon-backend-974u.onrender.com/static/4865.jpg" />

🎉 Final Notes

✔ Static image URLs never change
✔ Backend handles all matching + AI logic
✔ Dataset is persistent across deployments
✔ Upload photo endpoint is fully integrated
✔ Perfect for any frontend framework