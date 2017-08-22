# coding=utf-8
# TODO must
# remove exactly same code
# feed source length as sequence_length
# beam search
# decoder
# decoder test
# feed forward

# TODO hyperparams
# dropout
# bucketing
# optimizer

from __future__ import print_function
from __future__ import division

import os
import time
import math
from datetime import datetime
import logging

import msgpack as pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import code_loader
import code_model
import source_filter

tf.app.flags.DEFINE_float("learning_rate", 0.002, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 0, "Token vocabulary size. (0: set by data total vocab)")
tf.app.flags.DEFINE_integer("epoch", 0, "How many epochs (0: no limit)")
tf.app.flags.DEFINE_string("data_dir", "./code-data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./code-data", "Training directory.")
tf.app.flags.DEFINE_string("train_data_path", None, "Training data.")
tf.app.flags.DEFINE_string("dev_data_path", None, "Training data.")
tf.app.flags.DEFINE_string("out_tag", "", "A tag for certain output.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 0, "How many training steps to do per checkpoint. (0: same with epoch)")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# encoding 이 두개니까 버킷을 어떻게 잡을지 생각좀 해봐야
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


# _buckets = [(5, 15), (20, 50)]
# _buckets = [(15, 25), (25, 35)]


def read_data(train_id_set, max_size=None):
    data_set = [[] for _ in _buckets]
    counter = 0
    saved = 0
    for code in train_id_set:
        for line_idx in xrange(len(code) - 2):
            if max_size and counter >= max_size:
                break
            counter += 1

            source_ids = [code[line_idx], code[line_idx + 2]]
            target_ids = [code[line_idx + 1]]
            target_ids[0].append(code_loader.EOS_ID)

            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(source_ids[0]) < source_size and len(source_ids[1]) < source_size and len(target_ids[0]) < target_size:
                    saved += 1
                    data_set[bucket_id].append([source_ids, target_ids])
                    break

    print("  not saved: %d" % (counter - saved))

    print("  read data line total %d" % counter)
    return data_set


def create_model(session, forward_only, vocab_size=FLAGS.vocab_size):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = code_model.CodeModel(
        vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def get_perplexity(loss):
    return math.exp(float(loss)) if loss < 300 else float("inf")


def train():
    train_id_data, id_to_vocab, vocab_to_id = code_loader.prepare_data(FLAGS.data_dir, FLAGS.vocab_size, cache=True)

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False, vocab_size=(FLAGS.vocab_size if FLAGS.vocab_size != 0 else len(id_to_vocab)))

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
        # dev_set_path = FLAGS.train_dir + "/dev_set." + str(FLAGS.from_vocab_size) + "." + pickle.__name__
        train_set_path = FLAGS.train_dir + "/train_set.ids" + str(FLAGS.vocab_size) + ".ds" + str(FLAGS.max_train_data_size) + "." + pickle.__name__

        if not tf.gfile.Exists(train_set_path) or True:
            print("Reading training data (limit: %d)." % FLAGS.max_train_data_size)
            train_set = read_data(train_id_data, FLAGS.max_train_data_size)
            with tf.gfile.GFile(train_set_path, "w") as f:
                pickle.dump(train_set, f)
        else:
            print("Loading training data (limit: %d)." % FLAGS.max_train_data_size)
            with tf.gfile.GFile(train_set_path, mode="r") as f:
                train_set = pickle.load(f)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        print("Running the training loop")

        step_time, loss = 0.0, 0.0
        current_step = 0
        epoch_step = 0
        previous_losses = []

        out = "Error: not enough steps to run with checkpoint_step."
        test_out = ""

        steps_per_epoch = (int(train_total_size) // FLAGS.batch_size)
        steps_per_checkpoint = FLAGS.steps_per_checkpoint if FLAGS.steps_per_checkpoint != 0 else steps_per_epoch

        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs_front, encoder_inputs_back, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs_front, encoder_inputs_back, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            # print condition
            if current_step % 1 == 0:
                print("  global step %d learning rate %.4f step-time %.2f loss %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                (time.time() - start_time), step_loss, get_perplexity(step_loss)))

            # per checkpoint jobs
            if current_step % steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                out = "checkpoint: global step %d learning rate %.4f step-time %.2f loss %.2f perplexity %.2f" % (
                    model.global_step.eval(), model.learning_rate.eval(), step_time, loss, get_perplexity(loss))
                print(out)

                test_out = decode_tester(sess, model)

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt" + ".size." + str(FLAGS.size) + ".nl." + str(FLAGS.num_layers))
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0

            # escape if all epoch is done
            if current_step % steps_per_epoch == 0:
                epoch_step += 1
                print("epoch %d" % epoch_step)
                if FLAGS.epoch != 0 and epoch_step >= FLAGS.epoch:
                    with open(FLAGS.train_dir + "/result.ids" + str(FLAGS.vocab_size) + ".ds" + str(FLAGS.max_train_data_size), "a") as output_file:
                        output_file.write(
                            datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ": " + out + " // Tag: " + FLAGS.out_tag + "\n" +
                            "\ttest_out: " + test_out + "\n"
                        )
                    break


def decode_tester(sess, model):
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.%s" % (FLAGS.vocab_size, pickle.__name__))
    with gfile.GFile(vocab_path, mode="r") as vocab_file:
        id_to_vocab, vocab_to_id, vocab_freq = pickle.load(vocab_file)

    source = "print sum(map(int, raw_input().split()))"
    data = [{
        "source": source
    }]
    data = source_filter.filter_danger(data, "(import\s+os)|(from\s+os)|(shutil)")
    source_filter.remove_redundent_lines(data)
    data = source_filter.set_token(data)
    source_data = code_loader.data_to_tokens_list(data)

    id_data = []
    for src in source_data:
        id_source = [[code_loader._START_LINE_ID]]
        for line in src:
            id_line = [vocab_to_id.get(word[1], code_loader.UNK_ID) for word in line]
            id_source.append(id_line)
        id_source.append([code_loader._END_LINE_ID])
        id_data.append(id_source)

    bucket_id = -1
    data_set = [[] for _ in _buckets]
    for code in id_data:
        for line_idx in xrange(len(code) - 2):

            source_ids = [code[line_idx], code[line_idx + 2]]
            target_ids = [code[line_idx + 1]]
            target_ids[0].append(code_loader.EOS_ID)

            for idx, (source_size, target_size) in enumerate(_buckets):
                if len(source_ids[0]) < source_size and len(source_ids[1]) < source_size:
                    data_set[idx].append([source_ids, target_ids])
                    bucket_id = idx
                    break

    if bucket_id == -1:
        logging.warning("Sentence truncated: %s", source)
        return

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs_front, encoder_inputs_back, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs_front, encoder_inputs_back, decoder_inputs, target_weights, bucket_id, True)
    # _, _, output_logits = model.step(sess, encoder_inputs_front, encoder_inputs_back, decoder_inputs, target_weights, bucket_id, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if code_loader.EOS_ID in outputs:
        outputs = outputs[:outputs.index(code_loader.EOS_ID)]
        # Print out French sentence corresponding to outputs.
    vocab_output = " ".join([tf.compat.as_str(id_to_vocab[output]) for output in outputs])
    print("\toutput: " + vocab_output)
    return vocab_output


def main(_):
    # if FLAGS.self_test:
    #     self_test()
    train()


if __name__ == "__main__":
    tf.app.run()
