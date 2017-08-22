# coding=utf-8
from __future__ import print_function

import os

from tensorflow.python.platform import gfile
import msgpack as pickle

import source_filter

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_LINE = b"_START_LINE"
_END_LINE = b"_END_LINE"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _START_LINE, _END_LINE]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_START_LINE_ID = 4
_END_LINE_ID = 5


# Regular expressions used to tokenize.
# _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
# _DIGIT_RE = re.compile(br"\d")

def create_vocabulary(vocabulary_path, data, max_vocabulary_size, cache=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path) or not cache:
        print("Creating vocabulary %s" % (vocabulary_path))
        vocab_freq = {}
        # counter = 0
        for source in data:
            for idx, tokens in enumerate(source):
                for w in tokens:
                    word = w[1]
                    # word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab_freq:
                        vocab_freq[word] += 1
                    else:
                        vocab_freq[word] = 1

        vocab_list = _START_VOCAB + sorted(vocab_freq, key=vocab_freq.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        id_to_vocab = dict([(id, vocab) for (id, vocab) in enumerate(vocab_list)])
        vocab_to_id = dict([(vocab, id) for (id, vocab) in enumerate(vocab_list)])
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            pickle.dump([id_to_vocab, vocab_to_id, vocab_freq], vocab_file)

    else:
        with gfile.GFile(vocabulary_path, mode="r") as vocab_file:
            id_to_vocab, vocab_to_id, vocab_freq = pickle.load(vocab_file)

    return id_to_vocab, vocab_to_id, vocab_freq


def data_to_token_ids(source_data, id_data_path, id_to_vocab, vocab_to_id, cache=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(id_data_path) or not cache:
        print("Creating id tokenized data %s" % id_data_path)

        id_data = []
        for source in source_data:
            id_source = [[_START_LINE_ID]]
            for line in source:
                id_line = [vocab_to_id.get(word[1], UNK_ID) for word in line]
                id_source.append(id_line)
            id_source.append([_END_LINE_ID])
            id_data.append(id_source)

        with gfile.GFile(id_data_path, mode="w") as id_data_file:
            pickle.dump(id_data, id_data_file)

    else:
        with gfile.GFile(id_data_path, mode="r") as token_file:
            id_data = pickle.load(token_file)

    return id_data


def data_to_tokens_list(data):
    data_tokens = map(lambda d: d["token"], data)
    train_data = []
    for tokens in data_tokens:
        separated_token = []
        line_tokens = []
        for idx, token in enumerate(tokens):
            if token[0] != 4:
                line_tokens.append(token)

            if idx == len(tokens) - 1 or token[0] == 4:
                separated_token.append(line_tokens)
                line_tokens = []
        train_data.append(separated_token)
    return train_data


def prepare_data(
        data_dir,
        vocabulary_size,
        data_path="1000-6-2017-07-13-12:55:21.msgpack",
        cache=True
):
    if vocabulary_size == 0:
        actual_vocab_size = 100000  # big enough

    data = None
    with open(data_dir + "/" + data_path) as data_file:
        data = pickle.load(data_file)

    data = source_filter.filter_danger(data, "(import\s+os)|(from\s+os)|(shutil)")
    source_filter.remove_redundent_lines(data)
    data = source_filter.set_token(data)  # delete untokenizable sources
    correct_data = filter(lambda d: d["accurate"], data)

    source_data = data_to_tokens_list(correct_data)

    # Create vocabularies of the appropriate sizes.
    # vocab 을 따로 관리할 필요 없음. 같은 소스이므로.
    vocab_path = os.path.join(data_dir, "vocab%d.%s" % (vocabulary_size, pickle.__name__))
    id_to_vocab, vocab_to_id, vocab_freq = create_vocabulary(vocab_path, source_data, actual_vocab_size, cache)

    train_path = data_dir + "/" + data_path
    # Create token ids for the training data.
    train_ids_path = train_path + (".ids%d" % vocabulary_size)
    train_id_data = data_to_token_ids(source_data, train_ids_path, id_to_vocab, vocab_to_id, cache)

    # Create token ids for the development data.
    # to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
    # from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
    # data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
    # data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)
    #
    return train_id_data, id_to_vocab, vocab_to_id
