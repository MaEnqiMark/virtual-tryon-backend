DATA_ROOT = "/data"
TRANSPARENT_DIR = DATA_ROOT / "transparent"
TRANSPARENT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/transparent", StaticFiles(directory=str(TRANSPARENT_DIR)), name="transparent")


@app.post("/upload_transparent_zip")
async def upload_transparent_zip(file: UploadFile = File(...)):
    zip_path = DATA_ROOT / "transparent_upload.zip"

    # Save uploaded zip
    with zip_path.open("wb") as f:
        f.write(await file.read())

    # Extract into TRANSPARENT_DIR
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(TRANSPARENT_DIR)

    zip_path.unlink()

    return {"status": "ok", "message": "Transparent images uploaded"}
