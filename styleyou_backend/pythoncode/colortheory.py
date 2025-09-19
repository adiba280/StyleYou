# colortheory.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from fastapi import Form

def extract_skin_tone(image_bytes):
    """Extracts the dominant skin tone from an uploaded image."""
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define skin tone range
    lower_skin = np.array([0, 20, 40], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)

    pixels = skin.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # Remove black pixels

    if len(pixels) == 0:
        return None

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(pixels)
    counts = Counter(kmeans.labels_)
    dominant_rgb = kmeans.cluster_centers_[max(counts, key=counts.get)]

    return tuple(map(int, dominant_rgb))

def determine_undertone(rgb):
    """Determines whether the skin undertone is Warm, Cool, or Neutral."""
    r, g, b = rgb
    hsv_color = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv_color

    if (h >= 160 and h <= 300) or (b > r and g >= r and (b - r) > 10):
        return "Cool"
    elif (h < 35 or h > 330) or (r > g and g > b and (r - b) > 15):
        return "Warm"
    else:
        return "Neutral"

def get_suitable_colors(undertone):
    """Returns a list of suitable wardrobe colors based on undertone."""
    color_map = {
        "Warm": ["beige", "camel", "warm red", "orange", "olive", "mustard", "gold", "brown"],
        "Cool": ["emerald green", "sapphire blue", "cool pink", "silver", "purple", "cool gray"],
        "Neutral": ["soft white", "taupe", "rose", "true red", "teal", "medium gray"]
    }
    return color_map.get(undertone, [])

# main.py
from fastapi import FastAPI, UploadFile, File
import pandas as pd
from colortheory import extract_skin_tone, determine_undertone, get_suitable_colors

app = FastAPI()

# Load CSV files
dataset_csv = "filtered_styles.csv"  # Contains clothing details (IDs, colors, etc.)
urls_csv = "images.csv"  # Contains image URLs linked to clothing items

df_data = pd.read_csv(dataset_csv)
df_urls = pd.read_csv(urls_csv)

def recommend_items_from_dataset(csv_path, suitable_colors, gender):
    """Filters dataset for clothing items that match the recommended colors."""
    df_data = pd.read_csv(csv_path)  # Load dataset here

    if "baseColour" not in df_data.columns or "gender" not in df_data.columns:
        return []
    gender_map = {1: "Men", 2: "Women"}
    gender_str = gender_map.get(gender, None)
    if gender_str is None:
        return [] 

    suitable_colors = [color.lower() for color in suitable_colors]
    recommended_items = df_data[(df_data["baseColour"].str.lower().isin(suitable_colors)) & (df_data["gender"] == gender_str)]

    return recommended_items[["id", "articleType", "baseColour"]].to_dict(orient="records")
async def recommend_clothing(file: UploadFile = File(...), gender: int = Form(...)):
    """Receives an image, determines skin undertone, and returns clothing recommendations."""
    image_bytes = await file.read()
    dominant_rgb = extract_skin_tone(image_bytes)

    if dominant_rgb is None:
        return {"error": "No skin pixels detected"}

    undertone = determine_undertone(dominant_rgb)
    suitable_colors = get_suitable_colors(undertone)
    recommendations = recommend_items_from_dataset(dataset_csv, suitable_colors, gender)
    item_ids = [item["id"] for item in recommendations]
    image_urls = get_image_urls(item_ids)

    return {
        "undertone": undertone,
        "recommended_colors": suitable_colors,
        "recommended_items": recommendations,
        "image_urls": image_urls
    }
