# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

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
START_LINE_ID = 4
END_LINE_ID = 5


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
      data:
      max_vocabulary_size: limit on the size of the created vocabulary.
      cache:
    """
    if not gfile.Exists(vocabulary_path) or not cache:
        print("Creating vocabulary %s" % vocabulary_path)
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
            print("  total vocab %d max %d removed %d" % (len(vocab_list), max_vocabulary_size, len(vocab_list) - max_vocabulary_size))
            vocab_list = vocab_list[:max_vocabulary_size]
        else:
            print("  total vocab %d max %d" % (len(vocab_list), max_vocabulary_size))

        id_to_vocab = dict([(idx, vocab) for (idx, vocab) in enumerate(vocab_list)])
        vocab_to_id = dict([(vocab, idx) for (idx, vocab) in enumerate(vocab_list)])
        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
            pickle.dump([id_to_vocab, vocab_to_id, vocab_freq], vocab_file)

    else:
        with gfile.GFile(vocabulary_path, mode="r") as vocab_file:
            id_to_vocab, vocab_to_id, vocab_freq = pickle.load(vocab_file)

    return id_to_vocab, vocab_to_id, vocab_freq


def data_to_token_ids(source_data, id_data_path, vocab_to_id, cache=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      source_data:
      id_data_path:
      vocab_to_id:
      cache: Boolean;
    """
    if not gfile.Exists(id_data_path) or not cache:
        print("Creating id tokenized data %s" % id_data_path)

        id_data = []
        for source in source_data:
            id_source = [[START_LINE_ID]]
            for line in source:
                id_line = [vocab_to_id.get(word[1], UNK_ID) for word in line]
                id_source.append(id_line)
            id_source.append([END_LINE_ID])
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
        data_path,
        cache=True
):
    original_vocab_size = vocabulary_size
    if vocabulary_size == 0:
        vocabulary_size = 10000000  # big enough

    data = []
    with open(data_dir + "/" + data_path) as data_file:
        data = pickle.load(data_file)

    data = source_filter.filter_danger(data)
    source_filter.remove_redundent_newlines_and_set_line_length(data)
    data = source_filter.set_token(data)  # delete untokenizable sources
    data = filter(lambda elm: elm["accurate"], data)  # get only accurate

    # remove too long source.
    _ = source_filter.get_length_stat(data)
    data = source_filter.filter_lines_too_long(data)

    # remove too much tokens per line.
    _ = source_filter.get_token_stat(data)
    data = source_filter.filter_tokens_too_much(data)

    # remove too less freq tokens with sources.
    _, _, removed_data, _ = source_filter.get_token_freq_stat(data)
    freq_limit_data = []
    for idx, d in enumerate(data):
        if idx not in removed_data:
            freq_limit_data.append(d)

    data = freq_limit_data

    source_data = data_to_tokens_list(data)

    # Create vocabularies of the appropriate sizes.
    # vocab 을 따로 관리할 필요 없음. 같은 소스이므로.
    vocab_path = os.path.join(data_dir, "vocab%d.%s" % (original_vocab_size, pickle.__name__))
    id_to_vocab, vocab_to_id, vocab_freq = create_vocabulary(vocab_path, source_data, vocabulary_size, cache)

    train_path = data_dir + "/" + data_path
    # Create token ids for the training data.
    train_ids_path = train_path + (".ids%d" % original_vocab_size)
    train_id_data = data_to_token_ids(source_data, train_ids_path, vocab_to_id, cache)

    # Create token ids for the development data.
    # to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
    # from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
    # data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
    # data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)
    #
    return train_id_data, id_to_vocab, vocab_to_id
