from fastapi import FastAPI, UploadFile, File, Form,HTTPException
import shutil
import os
import pandas as pd
from recommender_3 import recommend_items
from colortheory import extract_skin_tone, determine_undertone, get_suitable_colors, recommend_items_from_dataset
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CSV that contains image URLs
url_csv_path = "images.csv"
df_urls = pd.read_csv(url_csv_path)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), gender: int = Form(...)):
    """Receives an uploaded image, processes it, and returns recommendations."""
    print(f"🟢 Received Gender: {gender}")
    
    image_bytes = await file.read()  # ✅ Read image as bytes
    print(f"✅ Image received: {file.filename}, size: {len(image_bytes)} bytes")

    # Run color theory processing
    dominant_rgb = extract_skin_tone(image_bytes)  # ✅ Pass bytes instead of file path
    print(f"🎨 Extracted RGB: {dominant_rgb}")
    if dominant_rgb is None:
        return {"error": "Could not detect skin tone"}

    undertone = determine_undertone(dominant_rgb)
    print(f"🩸 Determined undertone: {undertone}")

    suitable_colors = get_suitable_colors(undertone)
    print(f"🎭 Suitable colors: {suitable_colors}")

    # Get recommended items based on colors
    csv_path = "filtered_styles.csv"
    recommendations = recommend_items_from_dataset(csv_path, suitable_colors, gender)
    df_urls["id"] = df_urls["id"].str.replace(".jpg", "", regex=False)

    # Attach corresponding image URLs from images.csv
    for item in recommendations:
        image_id = str(item["id"])
        image_url = df_urls[df_urls["id"] == image_id]["link"].values
        item["image_url"] = image_url[0] if len(image_url) > 0 else None

    print(f"✅ Final recommendations: {recommendations}")

    return {
        "filename": file.filename,
        "undertone": undertone,
        "suitable_colors": suitable_colors,
        "recommendations": recommendations
    }

UPLOAD_DIR = "uploads"

# Create the directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/recommend/")
async def recommend_fashion(image: UploadFile = File(...), gender: str = Form(...)):
    """Receives an image and gender, returns recommended fashion items."""
    print(f"📥 Received Gender: {gender}")
    
    # Save uploaded image
    image_path = f"{UPLOAD_DIR}/{image.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    # Call recommendation function
    recommendations = recommend_items(image_path, gender)
    print(f"🟢 Debug: recommendations output -> {recommendations}, Type: {type(recommendations)}")
    df_urls["id"] = df_urls["id"].str.replace(".jpg", "", regex=False)
    
    # Attach corresponding image URLs from images.csv
    for item in recommendations:
        image_id = str(item["id"])
        image_url = df_urls[df_urls["id"] == image_id]["link"].values
        item["image_url"] = image_url[0] if len(image_url) > 0 else None

    print(f"✅ Final recommendations: {recommendations}")
    
    return {
        "filename": image.filename,
        "recommendations": recommendations
    }


df = pd.read_csv("images.csv")  # Ensure this is the correct path

# Load .npy feature embeddings
image_features = np.load("filtered_features.npy")  # Ensure this is the correct file path

# Create a mapping of image ID to its index in the numpy array
image_id_to_index = {df.iloc[i]["id"]: i for i in range(len(df))}

class LikedImagesRequest(BaseModel):
    liked_images: list

@app.post("/similar-liked/")
async def get_similar_liked_items(request: LikedImagesRequest):
    liked_images = request.liked_images
    print("Received Liked Images:", liked_images)  # Debugging

    liked_indices = []
    for image_url in liked_images:
        matched_row = df[df["link"] == image_url]  # Find image ID from URL
        if not matched_row.empty:
            image_id = matched_row.iloc[0]["id"]
            if image_id in image_id_to_index:
                liked_indices.append(image_id_to_index[image_id])

    if not liked_indices:
        return {"similar_images": []}  # No matches found

    # Compute similarity for all liked images
    liked_vectors = image_features[liked_indices]  # Extract embeddings
    similarities = cosine_similarity(liked_vectors, image_features)  # Compare with all images
    avg_similarity = similarities.mean(axis=0)  # Average similarity scores

    # Get top similar items (excluding the liked images themselves)
    top_indices = np.argsort(avg_similarity)[::-1][:10]  # Get top 10 most similar
    similar_image_urls = df.iloc[top_indices]["link"].tolist()  # Get URLs

    print("Returning Similar Images:", similar_image_urls)  # Debugging
    return {"similar_images": similar_image_urls}
