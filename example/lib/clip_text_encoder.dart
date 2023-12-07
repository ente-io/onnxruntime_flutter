import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime_example/clip_tokenizer.dart';



class ClipTextEncoder {
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  ClipTextEncoder() {
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
    const assetFileName = 'assets/models/clip-text-vit-32-float32-int32.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    try {
      _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
      print('text model loaded');
    } catch (e, s) {
      print('text model not loaded');
    }
  }

  infer() async {
    // final runOptions = OrtRunOptions();
    // final data = List.filled(1, Int64List.fromList(List.filled(77, 0)));
    // final inputOrt = OrtValueTensor.createTensorWithDataList(data, [1, 77]);
    // final inputs = {'input': inputOrt};
    // final outputs = _session?.run(runOptions, inputs);
    // print((outputs?[0]?.value as List<List<double>>)[0]);
    // inputOrt.release();
    // runOptions.release();
    // _session?.release();
    // final encoding = encodingForModel("gpt2");
    // final tokenIntegers = encoding.encode("tiktoken is great!");
    // print(tokenIntegers);
    // [ 49406, 24986, 17134, 533, 830, 256, 49407, 0, 0, 0, … ]

    var tokenizer = CLIPTokenizer();
    await tokenizer.init();
    print(tokenizer.encode("tiktoken is great!"));
  }
}
