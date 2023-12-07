import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime_example/record_manager.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:onnxruntime_example/model_type_test.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_example/utils.dart';
import 'package:onnxruntime_example/vad_iterator.dart';
import 'package:onnxruntime_example/clip_image_encoder.dart';
import 'package:onnxruntime_example/clip_text_encoder.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final imgPath = "assets/images/cycle.jpg";
  late String _version;
  String? _pcmPath;
  String? _wavPath;
  AudioPlayer? _audioPlayer;
  VadIterator? _vadIterator;
  ClipImageEncoder? _clipImageEncoder;
  ClipTextEncoder? _clipTextEncoder;
  static const frameSize = 64;

  @override
  void initState() {
    super.initState();
    _version = OrtEnv.version;

    _vadIterator = VadIterator(frameSize, RecordManager.sampleRate);
    _vadIterator?.initModel();

    _clipImageEncoder = ClipImageEncoder();
    _clipImageEncoder?.initModel();
    _clipTextEncoder = ClipTextEncoder();
    _clipTextEncoder?.initModel();
  }

  @override
  Widget build(BuildContext context) {
    const textStyle = TextStyle(fontSize: 16);
    return MaterialApp(
      theme: ThemeData(useMaterial3: true),
      home: Scaffold(
        appBar: AppBar(
          title: const Text('OnnxRuntime'),
          centerTitle: true,
        ),
        body: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'OnnxRuntime Version = $_version',
                  style: textStyle,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(
                  height: 50,
                ),
                FutureBuilder<String>(
                  future: getAccessiblePathForAsset(imgPath, "test.jpg"),
                  builder:
                      (BuildContext context, AsyncSnapshot<String> snapshot) {
                    if (snapshot.hasData) {
                      final file = File(snapshot.data!).readAsBytesSync();
                      final rgb = img.decodeJpg(file)!;
                      var inputImage = img.copyResize(rgb,
                          height: 224, interpolation: img.Interpolation.linear);
                      inputImage = img.copyCrop(
                        inputImage,
                        x: (inputImage.width - 224) ~/ 2,
                        y: 0,
                        width: 224,
                        height: 224,
                      );
                      return Column(
                        children: [
                          Image.memory(Uint8List.fromList(file)),
                          const Padding(padding: EdgeInsets.all(24)),
                          Image.memory(
                              Uint8List.fromList(img.encodeJpg(inputImage))),
                        ],
                      );
                    } else {
                      return SizedBox.shrink();
                    }
                  },
                ),
                TextButton(
                    onPressed: () async {
                      final audioSource = await RecordManager.instance.start();
                      _pcmPath = audioSource?[0];
                      _wavPath = audioSource?[1];
                    },
                    child: const Text('Start Recording')),
                const SizedBox(
                  height: 50,
                ),
                TextButton(
                    onPressed: () {
                      RecordManager.instance.stop();
                    },
                    child: const Text('Stop Recording')),
                const SizedBox(
                  height: 50,
                ),
                TextButton(
                    onPressed: () async {
                      _audioPlayer = AudioPlayer();
                      await _audioPlayer?.play(DeviceFileSource(_wavPath!));
                    },
                    child: const Text('Start Playing')),
                const SizedBox(
                  height: 50,
                ),
                TextButton(
                    onPressed: () {
                      _audioPlayer?.stop();
                    },
                    child: const Text('Stop Playing')),
                const SizedBox(
                  height: 50,
                ),
                TextButton(
                    onPressed: () {
                      infer();
                    },
                    child: const Text('Start Inferring')),
              ],
            ),
          ),
        ),
      ),
    );
  }

  infer() async {
    final startTime = DateTime.now().millisecondsSinceEpoch;
    // print('out=${(await ModelTypeTest.testBool())[0].value}');
    // print('out=${(await ModelTypeTest.testFloat())[0].value}');
    // print('out=${(await ModelTypeTest.testInt64())[0].value}');
    // print('out=${(await ModelTypeTest.testString())[0].value}');


    // final path = await getAccessiblePathForAsset(imgPath, "test2.jpg");
    // _clipImageEncoder?.inferByImage(path);

      
    final endTime = DateTime.now().millisecondsSinceEpoch;
    print('infer cost time=${endTime - startTime}ms');
    _clipTextEncoder?.infer();
  }

  Int16List _transformBuffer(List<int> buffer) {
    final bytes = Uint8List.fromList(buffer);
    return Int16List.view(bytes.buffer);
  }

  @override
  void dispose() {
    _vadIterator?.release();
    super.dispose();
  }
}
