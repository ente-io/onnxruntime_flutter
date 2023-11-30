import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_example/processed_image.dart';
import 'package:flutter/painting.dart' as paint show decodeImageFromList;

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
    const assetFileName = 'assets/models/clip_visual.onnx';
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

  inferByImage(String imagePath) async {
    final runOptions = OrtRunOptions();
    final startTime = DateTime.now();

    // Change this with path
    // Code from Satan
    // final rgb8 = img.Image(width: 784, height: 890, format: img.Format.float32);
    // final rgb = img.decodePng(File(imagePath).readAsBytesSync()) as img.Image;
    // final inputImage = img.copyResize(rgb,
    //     width: 224, height: 224, interpolation: img.Interpolation.linear);

    // Hard coded Input
    // final processedImage = getProcessedImage();

    final image =
        await paint.decodeImageFromList(File(imagePath).readAsBytesSync());
    final resizedImage =
        await resizeImage(image, 224, 224, maintainAspectRatio: true);
    final croppedImage = await cropImage(
      resizedImage.$1,
      x: 0,
      y: 0,
      width: 224,
      height: 224,
    );
    final mean = [0.48145466, 0.4578275, 0.40821073];
    final std = [0.26862954, 0.26130258, 0.27577711];
    final ByteData imgByteData = await getByteDataFromImage(image);
    final processedImage =
        imageToByteListFloat32(croppedImage, imgByteData, 224, mean, std);

    final inputOrt = OrtValueTensor.createTensorWithDataList(
        processedImage, [1, 3, 224, 224]);
    final inputs = {'input': inputOrt};
    final outputs = _session?.run(runOptions, inputs);
    final result = (outputs?[0]?.value as List<List<double>>)[0];
    final endTime = DateTime.now();
    print((endTime.millisecondsSinceEpoch - startTime.millisecondsSinceEpoch)
            .toString() +
        "ms");
    print(result.toString());
  }

  Future<(Image, Size)> resizeImage(
    Image image,
    int width,
    int height, {
    FilterQuality quality = FilterQuality.medium,
    bool maintainAspectRatio = false,
  }) async {
    if (image.width == width && image.height == height) {
      return (image, Size(width.toDouble(), height.toDouble()));
    }
    final recorder = PictureRecorder();
    final canvas = Canvas(
      recorder,
      Rect.fromPoints(
        const Offset(0, 0),
        Offset(width.toDouble(), height.toDouble()),
      ),
    );

    double scaleW = width / image.width;
    double scaleH = height / image.height;
    if (maintainAspectRatio) {
      final scale = min(width / image.width, height / image.height);
      scaleW = scale;
      scaleH = scale;
    }
    final scaledWidth = (image.width * scaleW).round();
    final scaledHeight = (image.height * scaleH).round();

    canvas.drawImageRect(
      image,
      Rect.fromPoints(
        const Offset(0, 0),
        Offset(image.width.toDouble(), image.height.toDouble()),
      ),
      Rect.fromPoints(
        const Offset(0, 0),
        Offset(scaledWidth.toDouble(), scaledHeight.toDouble()),
      ),
      Paint()..filterQuality = quality,
    );

    final picture = recorder.endRecording();
    final resizedImage = await picture.toImage(width, height);
    return (
      resizedImage,
      Size(scaledWidth.toDouble(), scaledHeight.toDouble())
    );
  }

  Float32List imageToByteListFloat32(
    Image image,
    ByteData data,
    int inputSize,
    List<double> mean,
    List<double> std,
  ) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    assert(mean.length == 3);
    assert(std.length == 3);

    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = readPixelColor(image, data, j, i);
        buffer[pixelIndex++] = (pixel.red - mean[0]) / std[0];
        buffer[pixelIndex++] = (pixel.green - mean[1]) / std[1];
        buffer[pixelIndex++] = (pixel.blue - mean[2]) / std[2];
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  Color readPixelColor(
    Image image,
    ByteData byteData,
    int x,
    int y,
  ) {
    if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
      // throw ArgumentError('Invalid pixel coordinates.');
      return const Color(0x00000000);
    }
    assert(byteData.lengthInBytes == 4 * image.width * image.height);

    final int byteOffset = 4 * (image.width * y + x);
    return Color(_rgbaToArgb(byteData.getUint32(byteOffset)));
  }

  int _rgbaToArgb(int rgbaColor) {
    final int a = rgbaColor & 0xFF;
    final int rgb = rgbaColor >> 8;
    return rgb + (a << 24);
  }

  Future<Image> cropImage(
    Image image, {
    required double x,
    required double y,
    required double width,
    required double height,
    Size? maxSize,
    Size? minSize,
    double rotation = 0.0, // rotation in radians
    FilterQuality quality = FilterQuality.medium,
  }) async {
    // Calculate the scale for resizing based on maxSize and minSize
    double scaleX = 1.0;
    double scaleY = 1.0;
    if (maxSize != null) {
      final minScale = min(maxSize.width / width, maxSize.height / height);
      if (minScale < 1.0) {
        scaleX = minScale;
        scaleY = minScale;
      }
    }
    if (minSize != null) {
      final maxScale = max(minSize.width / width, minSize.height / height);
      if (maxScale > 1.0) {
        scaleX = maxScale;
        scaleY = maxScale;
      }
    }

    // Calculate the final dimensions
    final targetWidth = (width * scaleX).round();
    final targetHeight = (height * scaleY).round();

    // Create the canvas
    final recorder = PictureRecorder();
    final canvas = Canvas(
      recorder,
      Rect.fromPoints(
        const Offset(0, 0),
        Offset(targetWidth.toDouble(), targetHeight.toDouble()),
      ),
    );

    // Apply rotation
    final center = Offset(targetWidth / 2, targetHeight / 2);
    canvas.translate(center.dx, center.dy);
    canvas.rotate(rotation);

    // Enlarge both the source and destination boxes to account for the rotation (i.e. avoid cropping the corners of the image)
    final List<double> enlargedSrc =
        getEnlargedAbsoluteBox([x, y, x + width, y + height], 1.5);
    final List<double> enlargedDst = getEnlargedAbsoluteBox(
      [
        -center.dx,
        -center.dy,
        -center.dx + targetWidth,
        -center.dy + targetHeight,
      ],
      1.5,
    );

    canvas.drawImageRect(
      image,
      Rect.fromPoints(
        Offset(enlargedSrc[0], enlargedSrc[1]),
        Offset(enlargedSrc[2], enlargedSrc[3]),
      ),
      Rect.fromPoints(
        Offset(enlargedDst[0], enlargedDst[1]),
        Offset(enlargedDst[2], enlargedDst[3]),
      ),
      Paint()..filterQuality = quality,
    );

    final picture = recorder.endRecording();

    return picture.toImage(targetWidth, targetHeight);
  }

  List<double> getEnlargedAbsoluteBox(List<double> box, [double factor = 2]) {
    final boxCopy = List<double>.from(box, growable: false);
    // The four values of the box in order are: [xMinBox, yMinBox, xMaxBox, yMaxBox].

    final width = boxCopy[2] - boxCopy[0];
    final height = boxCopy[3] - boxCopy[1];

    boxCopy[0] -= width * (factor - 1) / 2;
    boxCopy[1] -= height * (factor - 1) / 2;
    boxCopy[2] += width * (factor - 1) / 2;
    boxCopy[3] += height * (factor - 1) / 2;

    return boxCopy;
  }

  Future<ByteData> getByteDataFromImage(
    Image image, {
    ImageByteFormat format = ImageByteFormat.rawRgba,
  }) async {
    final ByteData? byteDataRgba = await image.toByteData(format: format);
    if (byteDataRgba == null) {
      throw Exception('Could not convert image to ByteData');
    }
    return byteDataRgba;
  }
}
