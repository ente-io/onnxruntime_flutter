import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

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
}
