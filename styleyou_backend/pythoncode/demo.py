import pandas as pd

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Example: Fetching image URLs from dataset
def get_image_url(image_id):
    row = df[df['id'] == image_id]
    return row['link'].values[0] if not row.empty else None

# Example usage:
print(get_image_url(1855))  # Replace with a real image ID
