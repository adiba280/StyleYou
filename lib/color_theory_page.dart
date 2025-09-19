import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'liked_items_screen.dart';

class ColorTheoryPage extends StatefulWidget {
  @override
  _ColorTheoryPageState createState() => _ColorTheoryPageState();
}

class _ColorTheoryPageState extends State<ColorTheoryPage> {
  File? _image;
  Uint8List? _imageBytes;
  List<String> suitableColors = [];
  List<String> recommendedImages = [];
  Set<String> likedImages = {};
  int _selectedGender = 1;

  @override
  void initState() {
    super.initState();
    _loadLikedImages();
  }

  Future<void> _loadLikedImages() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      likedImages = prefs.getStringList('likedImages')?.toSet() ?? {};
    });
  }

  Future<void> _toggleLike(String imageUrl) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    List<String> likedImages = prefs.getStringList('likedImages') ?? [];

    if (likedImages.contains(imageUrl)) {
      likedImages.remove(imageUrl);
    } else {
      likedImages.add(imageUrl);
    }

    await prefs.setStringList('likedImages', likedImages); // Save for ColorTheoryPage
    setState(() {});
  }



  Future<void> _pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      if (kIsWeb) {
        Uint8List bytes = await pickedFile.readAsBytes();
        setState(() {
          _imageBytes = bytes;
        });
      } else {
        setState(() {
          _image = File(pickedFile.path);
        });
      }
      _uploadImage(pickedFile);
    }
  }

  Future<void> _uploadImage(XFile imageFile) async {
    var request = http.MultipartRequest('POST', Uri.parse("http://127.0.0.1:8000/upload-image/"));
    if (kIsWeb) {
      Uint8List imageBytes = await imageFile.readAsBytes();
      request.files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: 'upload.jpg'));
    } else {
      request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));
    }
    request.fields['gender'] = _selectedGender.toString();

    var response = await request.send();
    var responseBody = await response.stream.bytesToString();
    var data = jsonDecode(responseBody);

    setState(() {
      suitableColors = List<String>.from(data["suitable_colors"]);
      recommendedImages = _shuffleAndBalanceImages(data["recommendations"]);
    });
  }

  List<String> _shuffleAndBalanceImages(List<dynamic> recommendations) {
    Map<String, List<String>> colorToImages = {};
    for (var item in recommendations) {
      String color = item["baseColour"].toString().toLowerCase();
      String imageUrl = item["image_url"] ?? "";
      if (imageUrl.isNotEmpty) {
        colorToImages.putIfAbsent(color, () => []).add(imageUrl);
      }
    }
    List<String> shuffledImages = colorToImages.values.expand((x) => x).toList();
    shuffledImages.shuffle();
    return shuffledImages;
  }

  Widget _buildImage() {
    if (_image == null && _imageBytes == null) {
      return CircleAvatar(radius: 60, backgroundColor: Colors.grey.shade300);
    }
    return kIsWeb
        ? CircleAvatar(radius: 60, backgroundImage: MemoryImage(_imageBytes!))
        : CircleAvatar(radius: 60, backgroundImage: FileImage(_image!));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Color Theory"),
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
      body: Column(
        children: [
          SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text("Select Gender:"),
              SizedBox(width: 10),
              DropdownButton<int>(
                value: _selectedGender,
                items: [
                  DropdownMenuItem(child: Text("Men"), value: 1),
                  DropdownMenuItem(child: Text("Women"), value: 2),
                ],
                onChanged: (value) {
                  if (value != null) {
                    setState(() {
                      _selectedGender = value;
                    });
                    if (_image != null || _imageBytes != null) {
                      _uploadImage(XFile(_image?.path ?? ''));
                    }
                  }
                },
              ),
            ],
          ),
          SizedBox(height: 20),
          GestureDetector(
            onTap: _pickImage,
            child: _buildImage(),
          ),
          SizedBox(height: 20),
          Text("Suitable Colors:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          Wrap(
            children: suitableColors.map((color) => Container(
              margin: EdgeInsets.all(5),
              width: 30,
              height: 30,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _getColorFromName(color),
              ),
            )).toList(),
          ),
          SizedBox(height: 20),
          Text("Recommended Items:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          Expanded(
            child: GridView.builder(
              padding: EdgeInsets.all(10),
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
                crossAxisSpacing: 10,
                mainAxisSpacing: 10,
                childAspectRatio: 1,
              ),
              itemCount: recommendedImages.length,
              itemBuilder: (context, index) {
                String imageUrl = recommendedImages[index];
                return Stack(
                  children: [
                    Positioned.fill(
                      child: Image.network(imageUrl, fit: BoxFit.cover),
                    ),
                    Positioned(
                      top: 5,
                      right: 5,
                      child: IconButton(
                        icon: Icon(
                          likedImages.contains(imageUrl) ? Icons.favorite : Icons.favorite_border,
                          color: likedImages.contains(imageUrl) ? Colors.red : Colors.white,
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
    );
  }
  Color _getColorFromName(String colorName) {
    return Colors.primaries[(colorName.hashCode % Colors.primaries.length)];
  }
}
