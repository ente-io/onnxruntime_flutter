import 'dart:convert';
import 'dart:core';
import 'dart:io';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:html_unescape/html_unescape.dart';
import 'package:tuple/tuple.dart';

class CLIPTokenizer {
  String bpePath = "assets/vocab/bpe_simple_vocab_16e6.txt";
  late Map<int, String> byteEncoder;
  late Map<String, int> byteDecoder;
  late Map<int, String> decoder;
  late Map<String, int> encoder;
  late Map<Tuple2<String, String>, int> bpeRanks;
  Map<String, String> cache = <String, String>{'<|startoftext|>':'<|startoftext|>', '<|endoftext|>':'<|endoftext|>'};
  RegExp pat = RegExp(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", caseSensitive: false, multiLine: false,);
  
  late int sot;
  late int eot;

  CLIPTokenizer() {
    //init();
    
  }

  Future init() async {
    
    final bpe = await loadFile();
    byteEncoder = bytesToUnicode();
    byteDecoder = byteEncoder.map((k, v) => MapEntry(v, k));

    var _merges = bpe.split('\n');
    _merges = _merges.sublist(1, 49152 - 256 - 2 + 1);
    var merges = _merges.map((merge) => Tuple2(merge.split(' ')[0], merge.split(' ')[1])).toList();

    var vocab = byteEncoder.values.toList();
    vocab.addAll(vocab.map((v) => '$v</w>').toList());

    for(var merge = 0; merge < merges.length; merge++) {
      vocab.add(merges[merge].item1 + merges[merge].item2);
    }
    vocab.addAll(['<|startoftext|>', '<|endoftext|>']);
    
    decoder = vocab.asMap();
    encoder = decoder.map((k, v) => MapEntry(v, k));
    bpeRanks = Map.fromIterables(
      merges.map((merge) => merge),
      List.generate(merges.length, (i) => i)
    );

    sot = encoder['<|startoftext|>']!;
    eot = encoder['<|endoftext|>']!;
  }

  Future<String> loadFile() async {
    return await rootBundle.loadString(bpePath);
  }

  List<int> encode(String text) {
    List<int> bpeTokens = [];
    text = whitespaceClean(basicClean(text)).toLowerCase();
    for (Match match in pat.allMatches(text)) {
      String token = match[0]!;
      token = utf8.encode(token).map((b) => byteEncoder[b]).join();
      bpe(token).split(' ').forEach((bpeToken) => bpeTokens.add(encoder[bpeToken]!));
    }
    return bpeTokens;
  }

  String bpe(String token) {
    if (cache.containsKey(token)) {
      return cache[token]!;
    }
    var word = token.split('').map((char) => char).toList();
    word[word.length - 1] = word.last + '</w>';
    var pairs = getPairs(word);
    if (pairs.isEmpty) {
      return token + '</w>';
    }

    while (true) {

      Tuple2<String, String> bigram = pairs.first;
      for (var pair in pairs) {
        var rank1 = bpeRanks[pair] ?? -double.infinity;
        var rank2 = bpeRanks[bigram] ?? -double.infinity;

        if (rank1 < rank2) {
          bigram = pair;
        }
      }
    
      if (!bpeRanks.containsKey(bigram)) {
        break;
      }
      var first = bigram.item1;
      var second = bigram.item2;
      var newWord = <String>[];
      var i = 0;
      while (i < word.length) {
        var j = word.sublist(i).indexOf(first);
        if (j == -1) {
          newWord.addAll(word.sublist(i));
          break;
        }
        newWord.addAll(word.sublist(i, i + j));
        i = i + j;
        if (word[i] == first && i < word.length - 1 && word[i + 1] == second) {
          newWord.add(first + second);
          i += 2;
        } else {
          newWord.add(word[i]);
          i += 1;
        }
      }

      word = newWord;
      if (word.length == 1) {
        break;
      } else {
        pairs = getPairs(word);
      }
    }
    var wordStr = word.join(' ');
    cache[token] = wordStr;
    return wordStr;
  }

  List<int> tokenize(String text, {int nText = 76, bool pad = true}) {
    var tokens = encode(text);
    tokens = [sot] + tokens.sublist(0, min(nText - 1, tokens.length)) + [eot];
    if (pad) {
      return tokens + List.filled(nText + 1 - tokens.length, 0);
    } else {
      return tokens;
    }
  }

  List<int> pad (List<int> x, int padLength){
    return x + List.filled(padLength - x.length, 0);
  }

  Map<int, String> bytesToUnicode() {
    List<int> bs = [];
    for (int i = '!'.codeUnitAt(0); i <= '~'.codeUnitAt(0); i++) {
      bs.add(i);
    }
    for (int i = '¡'.codeUnitAt(0); i <= '¬'.codeUnitAt(0); i++) {
      bs.add(i);
    }
    for (int i = '®'.codeUnitAt(0); i <= 'ÿ'.codeUnitAt(0); i++) {
      bs.add(i);
    }

    List<int> cs = List.from(bs);
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n += 1;
      }
    }

    List<String> ds = cs.map((n) => String.fromCharCode(n)).toList();
    return Map.fromIterables(bs, ds);
  }

  Set<Tuple2<String, String>> getPairs(List<String> word) {
    Set<Tuple2<String, String>> pairs = {};
    String prevChar = word[0];
    for (var i = 1; i < word.length; i++) {
      pairs.add(Tuple2(prevChar, word[i]));
      prevChar = word[i];
    }
    return pairs;
  }

  String basicClean(String text) {
    var unescape = HtmlUnescape();
    text = unescape.convert(unescape.convert(text));
    return text.trim();
  }
  
  String whitespaceClean(String text) {
    text = text.replaceAll(RegExp(r'\s+'), ' ');
    return text.trim();
  }


}
