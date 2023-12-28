import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite/tflite.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MaskDetectionApp(),
    );
  }
}

class MaskDetectionApp extends StatefulWidget {
  @override
  _MaskDetectionAppState createState() => _MaskDetectionAppState();
}

class _MaskDetectionAppState extends State<MaskDetectionApp> {
  late CameraController _cameraController;
  late List<CameraDescription> _cameras;
  bool _isMaskDetected = false;
  bool _isModelLoaded = false;
  bool _isFrontCamera = false;
  List<dynamic>? _recognitions; // Variable pour stocker les reconnaissances

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    _cameraController = CameraController(
      _isFrontCamera ? _cameras[1] : _cameras[0],
      ResolutionPreset.medium,
    );
    await _cameraController.initialize();
    _cameraController.startImageStream(_processCameraImage);
  }

  Future<void> _loadModel() async {
    try {
      await Tflite.loadModel(
        model: 'assets/mask_detector.tflite',
        labels: 'assets/mask_detection_labels.txt',
      );
      setState(() {
        _isModelLoaded = true;
      });
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  Future<void> _processCameraImage(CameraImage cameraImage) async {
    if (!_isModelLoaded) {
      return;
    }

    try {
      var recognitions = await Tflite.runModelOnFrame(
        bytesList: cameraImage.planes.map((plane) {
          return plane.bytes;
        }).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        imageMean: 127.5,
        imageStd: 127.5,
        rotation: 90,
        numResults: 2,
        threshold: 0.1,
        asynch: true,
      );

      setState(() {
        _recognitions = recognitions; // Mettez à jour la variable avec les reconnaissances actuelles
      });

      if (recognitions != null && recognitions.isNotEmpty) {
        setState(() {
          _isMaskDetected = recognitions[0]['label'] == 'WithMask';
        });
      }
    } catch (e) {
      print('Error during inference: $e');
    }
  }

  @override
  void dispose() {
    _cameraController.dispose();
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Mask Detection'),
        actions: [
          IconButton(
            icon: Icon(Icons.camera_alt),
            onPressed: () {
              setState(() {
                _isFrontCamera = !_isFrontCamera;
                _initializeCamera();
              });
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          Positioned(
            bottom: 20,
            left: 20,
            child: Container(
              padding: EdgeInsets.all(10),
              decoration: BoxDecoration(
                border: Border.all(
                  color: _isMaskDetected ? Colors.green : Colors.red,
                  width: 3.0,
                ),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _isMaskDetected ? 'Avec masque' : 'Sans masque',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Confiance : ${(_recognitions?[0]['confidence'] ?? 0.0).toStringAsFixed(2)}',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                    ),
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
