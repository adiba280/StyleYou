import 'package:flutter/material.dart';
import 'color_theory_page.dart'; // Import ColorTheoryPage
import 'complementary_rec.dart'; // Import RecommenderPage
import 'liked_items_screen.dart'; // Import Liked Items Page

void main() {
  runApp(StyleYouApp());
}

class StyleYouApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  void navigateTo(BuildContext context, String title) {
    if (title == "Color Theory") {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => ColorTheoryPage()),
      );
    } else if (title == "Customize Style") {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => FashionRecommenderScreen()),
      );
    } else if (title == "Liked Images") {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => LikedItemsScreen()),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("StyleYou", style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.black,
        centerTitle: true,
        foregroundColor: Colors.deepOrangeAccent,
      ),
      body: Column(
        children: [
          // Two Gesture Detectors with Equal Size
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              Expanded(
                child: GestureDetector(
                  onTap: () => navigateTo(context, "Customize Style"),
                  child: Container(
                    height: 250, // Adjusted height
                    decoration: BoxDecoration(
                      image: DecorationImage(
                        image: AssetImage("assets/img1.jpg"),
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                ),
              ),
              Expanded(
                child: GestureDetector(
                  onTap: () => navigateTo(context, "Color Theory"),
                  child: Container(
                    height: 250, // Adjusted height
                    decoration: BoxDecoration(
                      image: DecorationImage(
                        image: AssetImage("assets/img2.jpg"),
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),

          SizedBox(height: 20),

          // Third Gesture Detector (Full Width)
          GestureDetector(
            onTap: () => navigateTo(context, "Liked Images"),
            child: Container(
              width: double.infinity,
              height: 80, // Height for the full-width button
              color: Colors.red.withOpacity(0.8), // Background color
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.favorite, color: Colors.white, size: 40), // Bigger Heart Icon
                  SizedBox(width: 10),
                  Text(
                    "Liked Images",
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
