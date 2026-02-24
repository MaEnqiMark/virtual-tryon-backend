import cv2
import numpy as np

# ========================
# CONFIG — SET THIS ONLY
# ========================
CLOTH_PATH = "/Users/markma/Downloads/zalando-hd-resized/test/transparent_imgs/14358_00.png"        # Input PNG with transparency
OUTPUT_PATH = "/Users/markma/Downloads/14358_00.png"

# ========================
# EDGE CLEANING LOGIC
# ========================

def clean_cloth_edges(img, white_thresh=220, erode_size=2, blur_size=3):
    """
    Remove bright/white halo pixels along garment edges, preserve alpha shape.
    white_thresh: remove pixels brighter than this value on cloth edge
    """

    if img.shape[2] < 4:
        raise ValueError("Image must have an alpha channel (PNG with transparency)!")

    # Split channels
    b, g, r, a = cv2.split(img)

    # Mask where cloth exists (alpha > 0)
    fg = a > 0

    # Pixel brightness inside garment only
    brightness = 0.2126*r + 0.7152*g + 0.0722*b
    white_mask = (brightness > white_thresh) & fg

    # Reduce alpha on white edge pixels
    a_clean = a.copy()
    a_clean[white_mask] = 0

    # Smooth the cleaned edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    a_clean = cv2.erode(a_clean, kernel, iterations=1)
    a_clean = cv2.GaussianBlur(a_clean, (blur_size, blur_size), 0)

    # Reassemble
    cleaned = cv2.merge([b, g, r, a_clean])
    return cleaned


if __name__ == "__main__":
    img = cv2.imread(CLOTH_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {CLOTH_PATH}")

    cleaned = clean_cloth_edges(img)
    cv2.imwrite(OUTPUT_PATH, cleaned)

    print(f"\n✨ Cleaned cloth saved to: {OUTPUT_PATH}\n")
