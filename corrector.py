# coding=utf-8
import tensorflow as tf
import msgpack as pickle

import code_loader

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "token vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./code-data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./code-data", "Training directory.")
tf.app.flags.DEFINE_string("train_data_path", None, "Training data.")
tf.app.flags.DEFINE_string("dev_data_path", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# encoding 이 두개니까 버킷을 어떻게 잡을지 생각좀 해봐야
_buckets = [(5, 20), (20, 50)]


def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()


def train():
    train_id_data, id_to_vocab, vocab_to_id = code_loader.prepare_data(FLAGS.data_dir, FLAGS.vocab_size)

    with tf.Session as sess:
        # Read data into buckets and compute their sizes.
        # print("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
        dev_set_path = FLAGS.train_dir + '/dev_set.' + str(FLAGS.from_vocab_size) + '.' + pickle.__name__
        train_set_path = FLAGS.train_dir + '/train_set.ids' + str(FLAGS.from_vocab_size) + '.ds' + str(
            FLAGS.max_train_data_size) + '.' + pickle.__name__

        if not tf.gfile.Exists(dev_set_path) or True:
            print("Reading development data")
            dev_set = read_data(from_dev, to_dev)
            with tf.gfile.GFile(dev_set_path, mode='w') as f:
                pickle.dump(dev_set, f)
        else:
            print("Loading development")
            with tf.gfile.GFile(dev_set_path, mode='r') as f:
                dev_set = pickle.load(f)

        # print(h.heap())

        if not tf.gfile.Exists(train_set_path) or True:
            print("Reading training data (limit: %d)." % FLAGS.max_train_data_size)
            train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
            with tf.gfile.GFile(train_set_path, 'w') as f:
                pickle.dump(train_set, f)
        else:
            print("Loading training data (limit: %d)." % FLAGS.max_train_data_size)
            with tf.gfile.GFile(train_set_path, mode='r') as f:
                train_set = pickle.load(f)


def main(_):
    # if FLAGS.self_test:
    #     self_test()
    train()


if __name__ == '__main__':
    tf.app.run()
