const express = require("express");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const csv = require("csv-parser");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

// Multer setup for image upload
const upload = multer({ dest: "uploads/" });

// Load image URLs from CSV
let imageURLs = {};
fs.createReadStream("image_urls.csv")
  .pipe(csv())
  .on("data", (row) => {
    imageURLs[row.id] = row.image_url; // Assuming 'id' and 'image_url' columns exist
  })
  .on("end", () => console.log("Image URLs loaded."));

// 📌 Route to handle image upload & recommendation
app.post("/analyze", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No image uploaded." });

  try {
    const pythonResponse = await axios.post("http://127.0.0.1:5000/color-theory", {
      image_path: req.file.path
    });

    const { recommended_ids } = pythonResponse.data;

    // Map IDs to their respective image URLs
    const recommendedImages = recommended_ids.map((id) => ({
      id,
      url: imageURLs[id] || null
    })).filter(item => item.url !== null); // Remove items without URLs

    res.json({ recommendedImages });
  } catch (error) {
    res.status(500).json({ error: "Error processing image", details: error.message });
  }
});

// Start server
app.listen(5000, () => console.log("Server running on port 5000"));
