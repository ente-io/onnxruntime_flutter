import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

Future<String> getAccessiblePathForAsset(
    String assetPath, String tempName) async {
  final byteData = await rootBundle.load(assetPath);
  final tempDir = await getTemporaryDirectory();
  final file = await File('${tempDir.path}/$tempName')
      .writeAsBytes(byteData.buffer.asUint8List());
  return file.path;
}
