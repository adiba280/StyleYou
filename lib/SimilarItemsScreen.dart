import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class SimilarItemsScreen extends StatefulWidget {
  final List<String> likedItems; // ✅ Accept liked items

  SimilarItemsScreen({required this.likedItems});

  @override
  _SimilarItemsScreenState createState() => _SimilarItemsScreenState();
}

class _SimilarItemsScreenState extends State<SimilarItemsScreen> {
  List<String> _likedItems = [];
  List<String> _similarImages = [];

  @override
  void initState() {
    super.initState();
    _loadLikedItems();
  }

  Future<void> _loadLikedItems() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      _likedItems = prefs.getStringList('likedItems') ?? [];
    });

    if (_likedItems.isNotEmpty) {
      _fetchSimilarImages();
    }
  }

  Future<void> _fetchSimilarImages() async {
    print("Sending liked images: $_likedItems"); // Debugging print

    var response = await http.post(
      Uri.parse('http://127.0.0.1:8000/similar-liked/'),
      headers: {"Content-Type": "application/json"},
      body: json.encode({"liked_images": _likedItems}),
    );

    print("Response Status Code: ${response.statusCode}");
    print("Response Body: ${response.body}");

    if (response.statusCode == 200) {
      var jsonData = json.decode(response.body);
      setState(() {
        _similarImages = List<String>.from(jsonData['similar_images']);
      });

      print("Similar Images: $_similarImages"); // Debugging print
    } else {
      print("API Error: ${response.statusCode}");
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Similar to Liked Items")),
      body: Column(
        children: [
          Text("Your Liked Items:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          SizedBox(height: 10),
          _likedItems.isEmpty
              ? Center(child: Text("No liked items yet"))
              : SizedBox(
            height: 100,
            child: ListView(
              scrollDirection: Axis.horizontal,
              children: _likedItems.map((imageUrl) {
                return Padding(
                  padding: const EdgeInsets.all(4.0),
                  child: Image.network(imageUrl, width: 80, height: 80),
                );
              }).toList(),
            ),
          ),
          Divider(),
          Expanded(
            child: _similarImages.isEmpty
                ? Center(child: Text("No similar items found"))
                : GridView.builder(
              padding: EdgeInsets.all(10),
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
                crossAxisSpacing: 8,
                mainAxisSpacing: 8,
              ),
              itemCount: _similarImages.length,
              itemBuilder: (context, index) {
                return Image.network(
                  _similarImages[index],
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) =>
                      Icon(Icons.error, size: 50, color: Colors.red),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
