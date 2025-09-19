import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'liked_items_screen.dart';

class FashionRecommenderScreen extends StatefulWidget {
  @override
  _FashionRecommenderScreenState createState() => _FashionRecommenderScreenState();
}

class _FashionRecommenderScreenState extends State<FashionRecommenderScreen> {
  XFile? _imageFile;
  Uint8List? _imageBytes;
  String? _selectedGender;
  List<String> _recommendations = [];
  List<String> _likedItems = [];
  final ImagePicker _picker = ImagePicker();

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
  }

  Future<void> _saveLikedItems() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setStringList('likedItems', _likedItems);
  }

  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      final bytes = await pickedFile.readAsBytes();
      setState(() {
        _imageFile = pickedFile;
        _imageBytes = bytes;
      });
    }
  }

  Future<void> _getRecommendations() async {
    if (_imageBytes == null || _selectedGender == null) return;

    var request = http.MultipartRequest(
      'POST',
      Uri.parse('http://127.0.0.1:8000/recommend/'),
    );

    request.files.add(http.MultipartFile.fromBytes(
      'image',
      _imageBytes!,
      filename: _imageFile!.name,
    ));
    request.fields['gender'] = _selectedGender!;

    var response = await request.send();
    var responseData = await response.stream.bytesToString();

    if (response.statusCode == 200) {
      try {
        var jsonData = json.decode(responseData);
        setState(() {
          _recommendations = List<String>.from(
            jsonData['recommendations'].map((item) => item['image_url'] ?? ''),
          );
        });
      } catch (e) {
        print("JSON Parsing Error: $e");
      }
    } else {
      print("API Error: ${response.statusCode} - $responseData");
    }
  }

  void _toggleLike(String imageUrl) {
    setState(() {
      if (_likedItems.contains(imageUrl)) {
        _likedItems.remove(imageUrl);
      } else {
        _likedItems.add(imageUrl);
      }
      _saveLikedItems();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Fashion Recommender"),
        actions: [
          IconButton(
            icon: Icon(Icons.favorite, color: Colors.red),
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => LikedItemsScreen(),
                ),
              );
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            GestureDetector(
              onTap: _pickImage,
              child: Container(
                width: 150,
                height: 150,
                decoration: BoxDecoration(
                  border: Border.all(),
                  color: Colors.grey[300],
                ),
                child: _imageBytes == null
                    ? Icon(Icons.image, size: 50)
                    : Image.memory(_imageBytes!, fit: BoxFit.cover),
              ),
            ),
            SizedBox(height: 16),
            DropdownButton<String>(
              hint: Text("Select Gender"),
              value: _selectedGender,
              items: [
                DropdownMenuItem(value: "1", child: Text("Male")),
                DropdownMenuItem(value: "2", child: Text("Female")),
              ],
              onChanged: (value) {
                setState(() {
                  _selectedGender = value;
                });
              },
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: _getRecommendations,
              child: Text("Get Recommendations"),
            ),
            Expanded(
              child: _recommendations.isEmpty
                  ? Center(child: Text("No recommendations yet"))
                  : GridView.builder(
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3,
                  crossAxisSpacing: 8,
                  mainAxisSpacing: 8,
                ),
                itemCount: _recommendations.length,
                itemBuilder: (context, index) {
                  String imageUrl = _recommendations[index];
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
                          icon: Icon(
                            _likedItems.contains(imageUrl)
                                ? Icons.favorite
                                : Icons.favorite_border,
                            color: Colors.red,
                          ),
                          onPressed: () => _toggleLike(imageUrl),
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
