import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;

class ClipImageEncoder {
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  ClipImageEncoder() {
    OrtEnv.instance.init();
    OrtEnv.instance.availableProviders().forEach((element) {
      print('onnx provider=$element');
    });
  }

  release() {
    _sessionOptions?.release();
    _sessionOptions = null;
    _session?.release();
    _session = null;
    OrtEnv.instance.release();
  }

  initModel() async {
    _sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
    const assetFileName = 'assets/models/visual.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    try {
      _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
      print('image model loaded');
    } catch (e, s) {
      print('image model not loaded');
    }
  }

  infer() async {
    final runOptions = OrtRunOptions();
    final data =
        List.filled(1, Float32List.fromList(List.filled(224 * 224 * 3, 0)));
    final inputOrt =
        OrtValueTensor.createTensorWithDataList(data, [1, 3, 224, 224]);
    final inputs = {'input': inputOrt};
    final outputs = _session?.run(runOptions, inputs);
    print((outputs?[0]?.value as List<List<double>>)[0]);
    inputOrt.release();
    runOptions.release();
    _session?.release();
  }

  inferByImage(String imagePath) {
    final runOptions = OrtRunOptions();

    // Change this with path
    //final rgb8 = img.Image(width: 784, height: 890, format: img.Format.float32);
    final rgb = img.decodeJpg(File(imagePath).readAsBytesSync()) as img.Image;
    final inputImage = img.copyResize(rgb,
        width: 224, height: 224, interpolation: img.Interpolation.linear);
    final mean = [0.48145466, 0.4578275, 0.40821073];
    final std = [0.26862954, 0.26130258, 0.27577711];
    final processedImage = imageToByteListFloat32(inputImage, 224, mean, std);

    final inputOrt = OrtValueTensor.createTensorWithDataList(
        processedImage, [1, 3, 224, 224]);
    final inputs = {'input': inputOrt};
    final outputs = _session?.run(runOptions, inputs);
    print((outputs?[0]?.value as List<List<double>>)[0]);
  }

  Float32List imageToByteListFloat32(
      img.Image image, int inputSize, List<double> mean, List<double> std) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    assert(mean.length == 3);
    assert(std.length == 3);

    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (pixel.r - mean[0]) / std[0];
        buffer[pixelIndex++] = (pixel.g - mean[1]) / std[1];
        buffer[pixelIndex++] = (pixel.b - mean[2]) / std[2];
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }
}
