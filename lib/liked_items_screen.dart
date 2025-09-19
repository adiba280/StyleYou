import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'SimilarItemsScreen.dart'; // Import the similar items screen

class LikedItemsScreen extends StatefulWidget {
  @override
  _LikedItemsScreenState createState() => _LikedItemsScreenState();
}

class _LikedItemsScreenState extends State<LikedItemsScreen> {
  List<String> _likedItems = [];

  @override
  void initState() {
    super.initState();
    _loadLikedItems();
  }

  Future<void> _loadLikedItems() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    List<String> colorTheoryLiked = prefs.getStringList('likedImages') ?? []; // From Color Theory Page
    List<String> fashionRecommenderLiked = prefs.getStringList('likedItems') ?? []; // From Fashion Recommender

    setState(() {
      _likedItems = {...colorTheoryLiked, ...fashionRecommenderLiked}.toList();
    });
  }

  Future<void> _removeLikedItem(String imageUrl) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      _likedItems.remove(imageUrl);
      prefs.setStringList('likedItems', _likedItems);
    });
  }

  @override
  Widget build(BuildContext context) {
    print("Liked Items Length: ${_likedItems.length}");  // Debugging print

    return Scaffold(
      appBar: AppBar(title: Text("Liked Items")),
      body: Column(
        children: [
          Expanded(
            child: _likedItems.isEmpty
                ? Center(child: Text("No liked items yet"))
                : GridView.builder(
              padding: EdgeInsets.all(10),
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
                crossAxisSpacing: 8,
                mainAxisSpacing: 8,
              ),
              itemCount: _likedItems.length,
              itemBuilder: (context, index) {
                String imageUrl = _likedItems[index];
                return Stack(
                  children: [
                    Image.network(
                      imageUrl,
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stackTrace) =>
                          Icon(Icons.error, size: 50, color: Colors.red),
                    ),
                    Positioned(
                      top: 5,
                      right: 5,
                      child: IconButton(
                        icon: Icon(Icons.delete, color: Colors.red),
                        onPressed: () => _removeLikedItem(imageUrl),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(10.0),
            child: ElevatedButton(
              onPressed: _likedItems.isEmpty
                  ? null // Disable if no liked items
                  : () {
                print("Navigating to SimilarItemsScreen with ${_likedItems.length} liked items"); // Debug
                print("Liked Items in Flutter: $_likedItems");

                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => SimilarItemsScreen(likedItems: _likedItems),
                  ),
                );
              },
              child: Text("Find Similar Items"),
            ),
          ),
        ],
      ),
    );
  }

}
