import 'dart:io';

class Tokenizer {
  String vocabPath = "";

  Tokenizer(path) {
    if (Directory(path).existsSync()) {
      vocabPath = path;
    }
  }
}
