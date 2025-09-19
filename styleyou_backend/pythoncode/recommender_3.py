import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
from collections import Counter
from sklearn.cluster import KMeans
from collections import Counter

# ================================
# Load CSV and Pre-Trained Model
# ================================
csv_file = "filtered_styles.csv"
df = pd.read_csv(csv_file)
urls_csv = "images.csv"
df_urls = pd.read_csv(urls_csv)

# Create label mapping for category classification
label_map = {label: idx for idx, label in enumerate(df['articleType'].unique())}
label_map_inv = {v: k for k, v in label_map.items()}  # Reverse mapping

# Load trained ResNet model for category classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(label_map))  # Adjust output layer
model.load_state_dict(torch.load("clothing_model.pth", map_location=device))
model.to(device)
model.eval()

print(f"✅ Model loaded with {len(label_map)} categories!")

# ================================
# Define Complementary Color Matching
# ================================
fashion_color_guide = {
    "Navy Blue": ["White", "Beige", "Gold", "Peach"],
    "Blue": ["White", "Tan", "Grey", "Yellow", "Brown"],
    "Silver": ["Black", "White", "Navy Blue", "Burgundy"],
    "Black": ["White", "Red", "Grey", "Beige", "Gold"],
    "Grey": ["Black", "White", "Pink", "Burgundy", "Teal"],
    "Green": ["White", "Beige", "Navy Blue", "Gold", "Brown"],
    "Purple": ["White", "Grey", "Gold", "Silver"],
    "White": ["Black", "Blue", "Red", "Olive", "Burgundy"],
    "Beige": ["Brown", "Olive", "Maroon", "Black"],
    "Brown": ["Cream", "White", "Green", "Beige"],
    "Bronze": ["White", "Black", "Gold", "Olive"],
    "Teal": ["White", "Grey", "Brown", "Gold"],
    "Copper": ["Black", "White", "Beige", "Brown"],
    "Pink": ["White", "Grey", "Navy Blue", "Gold"],
    "Off White": ["Black", "Grey", "Blue", "Burgundy"],
    "Maroon": ["Beige", "Gold", "White", "Navy Blue"],
    "Red": ["White", "Black", "Blue", "Beige"],
    "Khaki": ["Black", "White", "Brown", "Olive"],
    "Orange": ["Blue", "White", "Black", "Navy Blue"],
    "Coffee Brown": ["White", "Beige", "Black", "Olive"],
    "Yellow": ["Navy Blue", "Black", "Grey", "Brown"],
    "Charcoal": ["White", "Beige", "Black", "Silver"],
    "Gold": ["Black", "Navy Blue", "White", "Burgundy"],
    "Steel": ["Black", "White", "Navy Blue", "Silver"],
    "Tan": ["Black", "White", "Navy Blue", "Burgundy"],
    "Multi": ["Black", "White", "Blue", "Beige"],
    "Magenta": ["White", "Black", "Grey", "Gold"],
    "Lavender": ["White", "Grey", "Gold", "Silver"],
    "Sea Green": ["White", "Black", "Beige", "Brown"],
    "Cream": ["Brown", "Black", "Burgundy", "Navy Blue"],
    "Peach": ["White", "Navy Blue", "Gold", "Beige"],
    "Olive": ["White", "Black", "Beige", "Brown"],
    "Skin": ["Black", "White", "Olive", "Brown"],
    "Burgundy": ["White", "Beige", "Gold", "Navy Blue"],
    "Grey Melange": ["Black", "White", "Silver", "Burgundy"],
    "Rust": ["White", "Black", "Gold", "Brown"],
    "Rose": ["White", "Grey", "Burgundy", "Gold"],
    "Lime Green": ["White", "Black", "Brown", "Beige"],
    "Mauve": ["White", "Grey", "Gold", "Silver"],
    "Turquoise Blue": ["White", "Black", "Beige", "Gold"],
    "Metallic": ["Black", "White", "Navy Blue", "Silver"],
    "Mustard": ["Black", "Navy Blue", "White", "Beige"],
    "Taupe": ["Black", "White", "Burgundy", "Grey"],
    "Nude": ["Black", "White", "Olive", "Beige"],
    "Mushroom Brown": ["Black", "White", "Olive", "Beige"],
    "Fluorescent Green": ["Black", "White", "Grey", "Navy Blue"]
}
def remove_background(image):
    """Removes background using thresholding and contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255  # Start with a white mask
    cv2.drawContours(mask, contours, -1, 0, thickness=cv2.FILLED)  # Fill non-clothing areas
    return cv2.bitwise_and(image, image, mask=~mask)


# ================================
# Function to Extract Dominant Color
# ================================
def get_dominant_color(image_path, k=3):
    """Extracts the dominant color while avoiding the background (even if it's not white)."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Use dilation & closing to get a more solid object region
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours and get the largest one (assumed to be clothing)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (128, 128, 128)  # Default to neutral gray if no object found

    # Create a mask for the largest contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, (255), thickness=cv2.FILLED)

    # Apply mask to extract clothing pixels
    clothing_pixels = image[mask == 255]

    if len(clothing_pixels) == 0:
        return (128, 128, 128)  # Return gray if no valid clothing pixels

    # KMeans clustering to find dominant color
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(clothing_pixels)
    counts = Counter(labels)
    dominant_color = kmeans.cluster_centers_[max(counts, key=counts.get)]

    return tuple(map(int, dominant_color))



def get_closest_color(rgb):
    """Finds the closest fashion color name."""
    predefined_colors = {
    "Navy Blue": (0, 0, 128), "Blue": (0, 0, 255), "Silver": (192, 192, 192), "Black": (0, 0, 0),
    "Grey": (128, 128, 128), "Green": (0, 128, 0), "Purple": (128, 0, 128), "White": (255, 255, 255),
    "Beige": (245, 245, 220), "Brown": (139, 69, 19), "Bronze": (205, 127, 50), "Teal": (0, 128, 128),
    "Copper": (184, 115, 51), "Pink": (255, 192, 203), "Off White": (250, 250, 250), "Maroon": (128, 0, 0),
    "Red": (255, 0, 0), "Khaki": (195, 176, 145), "Orange": (255, 165, 0), "Coffee Brown": (111, 78, 55),
    "Yellow": (255, 255, 0), "Charcoal": (54, 69, 79), "Gold": (255, 215, 0), "Steel": (176, 196, 222),
    "Tan": (210, 180, 140), "Multi": (255, 255, 255), "Magenta": (255, 0, 255), "Lavender": (230, 230, 250),
    "Sea Green": (46, 139, 87), "Cream": (255, 253, 208), "Peach": (255, 218, 185), "Olive": (128, 128, 0),
    "Skin": (255, 224, 189), "Burgundy": (128, 0, 32), "Grey Melange": (169, 169, 169), "Rust": (183, 65, 14),
    "Rose": (255, 228, 225), "Lime Green": (50, 205, 50), "Mauve": (224, 176, 255), "Turquoise Blue": (0, 206, 209),
    "Metallic": (169, 169, 169), "Mustard": (255, 219, 88), "Taupe": (72, 60, 50), "Nude": (255, 204, 153),
    "Mushroom Brown": (150, 120, 110), "Fluorescent Green": (35, 255, 20)
}
    return min(predefined_colors.keys(), key=lambda c: np.linalg.norm(np.array(predefined_colors[c]) - np.array(rgb)))

# ================================
# Function to Recommend Items
# ================================
def recommend_items(image_path, user_gender):
    """Predicts category and color, then recommends complementary items based on gender."""
    # Step 1: Predict Category
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    predicted_label = label_map_inv.get(predicted_class, "Unknown")

    # Step 2: Detect Color
    dominant_rgb = get_dominant_color(image_path)
    detected_color = get_closest_color(dominant_rgb)
    complementary_colors = fashion_color_guide.get(detected_color, ["Black", "White"])
    gender_map = {1: "Men", 2: "Women"}
    user_gender_str = gender_map.get(user_gender,"Women")  # Default to "Men" if invalid input
    filtered_df = df[df["gender"] == user_gender_str]
    
    # Step 4: Recommend Matching Items with Balanced Categories
    category_mapping = {
    "Tops": ["Trousers", "Jeans", "Skirts", "Leggings", "Jackets", "Handbags", "Watches", "Earrings", "Necklaces", "Bracelets", "Rings"],
    "Shirts": ["Trousers", "Jeans", "Jackets", "Watches", "Belts"],
    "Tshirts": ["Jeans", "Shorts", "Casual Shoes", "Watches"],
    "Dresses": ["Heels", "Sandals", "Clutches", "Handbags", "Earrings", "Necklaces", "Bracelets", "Rings"],
    "Jumpsuit": ["Heels", "Handbags", "Earrings"],
    "Trousers": ["Shirts", "Tops", "Sweatshirts", "Belts"],
    "Jeans": ["Tops", "Shirts", "Tshirts", "Watches", "Belts"],
    "Skirts": ["Tops", "Tshirts", "Sweatshirts"],
    "Leggings": ["Kurtas", "Tops", "Tunics"],
    "Kurtas": ["Leggings", "Salwar", "Dupatta", "Heels", "Earrings"],
    "Sweatshirts": ["Jeans", "Trousers", "Casual Shoes"],
    "Jackets": ["Trousers", "Jeans", "Watches"],
    "Shoes": ["Jeans", "Trousers", "Dresses"],
    "Jewellery Set": ["Tops", "Dresses"],
    "Handbags": ["Tops", "Dresses", "Jeans"],
    "Belts": ["Trousers", "Jeans","Shirts"]
}
    recommended_items = filtered_df[
        (filtered_df["baseColour"].isin(complementary_colors)) &
        (filtered_df["articleType"].isin(category_mapping.get(predicted_label, [])))
    ]
    
    # Step 5: Balance categories in output
    category_counts = Counter(recommended_items["articleType"])
    balanced_recommendations = []
    for category in category_mapping.get(predicted_label, []):
        items = recommended_items[recommended_items["articleType"] == category]
        if not items.empty:
            balanced_recommendations.append(items.sample(min(10, len(items))))
    final_recommendations = (
        pd.concat(balanced_recommendations)
        if balanced_recommendations
        else recommended_items.sample(min(2675, len(recommended_items)))
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Display Results
    print("\n🎯 **Recommendation Results:**")
    print(f"👕 Predicted Category: {predicted_label}")
    print(f"🎨 Detected Color: {detected_color}")
    print(f"✅ Complementary Colors: {complementary_colors}")
    print("\n📌 **Recommended Items:**")
    print(final_recommendations[["id", "articleType", "baseColour"]])
    

    # Step 7: Convert to JSON
    output = {
        "category": predicted_label,
        "color": detected_color,
        "recommendations": final_recommendations[["id", "articleType", "baseColour"]].to_dict(orient="records")
    }

    return final_recommendations[["id", "articleType", "baseColour"]].to_dict(orient="records")




