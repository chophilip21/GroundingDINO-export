import collections
import unicodedata
import numpy as np

class BertTokenizer:
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}
        self.do_lower_case = do_lower_case
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.mask_token_id = self.vocab[self.mask_token]

    def decode(self, token_ids):
        tokens = [self.ids_to_tokens.get(token_id, self.unk_token) for token_id in token_ids]
        # Clean up tokens
        text = ' '.join(tokens)
        text = text.replace(' ##', '')
        text = text.replace('##', '')
        text = text.strip()
        return text

    def load_vocab(self, vocab_file):
        vocab = collections.OrderedDict()
        with open(vocab_file, 'r', encoding='utf-8') as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip('\n')
            vocab[token] = index
        return vocab

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

class BasicTokenizer:
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        tokens = self._whitespace_tokenize(text)
        split_tokens = []
        for token in tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._strip_accents(token)
            split_tokens.extend(self._split_on_punc(token))
        return self._whitespace_tokenize(' '.join(split_tokens))

    def _clean_text(self, text):
        return ''.join(c if not _is_control(c) else ' ' for c in text)

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            if _is_chinese_char(ord(char)):
                output.extend([' ', char, ' '])
            else:
                output.append(char)
        return ''.join(output)

    def _strip_accents(self, text):
        text = unicodedata.normalize('NFD', text)
        return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    def _split_on_punc(self, text):
        chars = list(text)
        tokens = []
        i = 0
        start_new_word = True
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                tokens.append(char)
                start_new_word = True
            else:
                if start_new_word:
                    tokens.append('')
                tokens[-1] += char
                start_new_word = False
            i += 1
        return tokens

    def _whitespace_tokenize(self, text):
        return text.strip().split()

class WordpieceTokenizer:
    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]
        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = ''.join(chars[start:end])
                if start > 0:
                    substr = '##' + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            start = end
        if is_bad:
            return [self.unk_token]
        else:
            return sub_tokens

def _is_control(char):
    return unicodedata.category(char) in ('Cc', 'Cf')

def _is_whitespace(char):
    return char in (' ', '\t', '\n', '\r') or unicodedata.category(char) == 'Zs'

def _is_punctuation(char):
    cp = ord(char)
    return (
        33 <= cp <= 47 or
        58 <= cp <= 64 or
        91 <= cp <= 96 or
        123 <= cp <= 126 or
        unicodedata.category(char).startswith('P')
    )

def _is_chinese_char(cp):
    return (
        0x4E00 <= cp <= 0x9FFF or
        0x3400 <= cp <= 0x4DBF or
        0x20000 <= cp <= 0x2A6DF or
        0x2A700 <= cp <= 0x2B73F or
        0x2B740 <= cp <= 0x2B81F or
        0x2B820 <= cp <= 0x2CEAF or
        0xF900 <= cp <= 0xFAFF or
        0x2F800 <= cp <= 0x2FA1F
    )