# coding=utf-8
# use output to guide the decoder
# use bag of word for state vector
import numpy as np
import tensorflow as tf
import data_gen
from utils import *

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


# idx (a batch of ints, [[5,2,4], ... ]
# num_labels: num_toks basically
# bbatch_size: either genbatchsize or trainbatchsize
# output a batch of [[[0,0,0,0,0,1], [0,0,1,0,0,0], [0,0,0,0,1,0]] i.e.
# output the 1 hot version of the idx
def idx_to_sym(idx, num_labels, bbatch_size):
    label_batch = idx
    sparse_labels = tf.reshape(label_batch, [-1, 1])
    indices = tf.reshape(tf.range(0, bbatch_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([bbatch_size, num_labels])
    labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    labels = tf.reshape(labels, [bbatch_size, num_labels])
    return labels


class NN_SANDWICH_BEAM:
    # the nn is intialized with these numerical constants
    # the nn itself knows nothing about the token representation it's just a numerical unit
    def __init__(self, stat, nn_config):

        # use the stat and config to set-up its parameters
        self.max_line_l = stat["bnd_seq_l"]
        self.n_toks = len(stat["bnd_toks"])

        self.batch_size = nn_config["batch_size"]
        self.encoder_units = nn_config["encoder_units"]
        self.decoder_units = nn_config["decoder_units"]
        self.dec_state_size = 3 * self.decoder_units + self.n_toks
        self.beam_batch_size = nn_config["beam_batch_size"]
        self.beam_size = nn_config["beam_size"]

        self.sess = tf.Session()

        # ========== encoder ===========
        self.lstm_encode = tf.nn.rnn_cell.MultiRNNCell(
            cells=[
                tf.nn.rnn_cell.LSTMCell(
                    num_units=self.encoder_units,
                    # input_size=self.n_toks,
                    initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1, seed=None)
                ),
                tf.nn.rnn_cell.LSTMCell(
                    num_units=self.encoder_units,
                    # input_size=self.encoder_units,
                    initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1, seed=None)
                )
            ]
        )

        # ========== duct tape ===========
        # i have 2 things of different size, put a matrix in the middle to glue them
        self.duct_tape = tf.Variable(
            tf.random_normal(
                [self.encoder_units * 8, self.dec_state_size],
                mean=0.1, stddev=0.1, dtype=tf.float32
            ),
            name="tape"
        )

        # ========== decoder ==========
        self.lstm_decode = tf.nn.rnn_cell.MultiRNNCell(
            cells=[
                tf.nn.rnn_cell.LSTMCell(
                    num_units=self.decoder_units,
                    # input_size=self.n_toks,
                    initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1, seed=None)
                ),
                tf.nn.rnn_cell.LSTMCell(
                    num_units=self.decoder_units,
                    # input_size=self.decoder_units,
                    initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1, seed=None),
                    num_proj=self.n_toks
                )
            ]
        )

    def generate_train_graph(self, model_loc=None):
        # I don't know why these are here but LOL!
        max_line_l = self.max_line_l
        n_toks = self.n_toks

        batch_size = self.batch_size
        # encoder_units = self.encoder_units
        # decoder_units = self.decoder_units
        # dec_state_size = self.dec_state_size
        # beam_batch_size = self.beam_batch_size
        # beam_size = self.beam_size

        # let's first set up all the inputs and output...
        self.data_front = tf.placeholder(tf.float32, shape=(batch_size, max_line_l, n_toks))
        self.data_front_l = tf.placeholder(tf.int32, shape=(batch_size))
        self.data_back = tf.placeholder(tf.float32, shape=(batch_size, max_line_l, n_toks))
        self.data_back_l = tf.placeholder(tf.int32, shape=(batch_size))
        self.data_out = tf.placeholder(tf.float32, shape=(batch_size, max_line_l, n_toks))

        # encoding
        # listify the front and back inputs
        lst_front = [self.data_front[:, j, :] for j in range(max_line_l)]
        lst_back = [self.data_back[:, j, :] for j in range(max_line_l)]

        # compress both front and back with lstm
        #    with tf.variable_scope("encoding") as scope:
        # tf.nn.rnn 에서 tf.contrib.rnn 으로 바뀜. tf.contrib.rnn == tf.nn.dynamic_rnn
        _, front_state = tf.contrib.rnn(
            cell=self.lstm_encode,
            inputs=lst_front,
            dtype=tf.float32,
            sequence_length=self.data_front_l,
            scope="enc_front"
        )
        #    scope.reuse_variables()
        _, back_state = tf.contrib.rnn(
            cell=self.lstm_encode,
            inputs=lst_back,
            dtype=tf.float32,
            sequence_length=self.data_back_l,
            scope="enc_back"
        )

        # concatenate the compressed front and back
        front_back = array_ops.concat(1, [front_state, back_state])

        # duct-tape the states to right dimensions
        decode_init_state = tf.matmul(front_back, self.duct_tape)

        # the output for generation is guided by this during traing
        lst_guide = [self.data_out[:, j, :] for j in range(max_line_l)]
        print "guide line shape ", show_dim(lst_guide)
        # tf.nn.seq2seq.rnn_decoder 에서 tf.contrib.legacy_seq2seq.rnn_decoder 로
        pred_line, state = tf.contrib.legacy_seq2seq.rnn_decoder(
            lst_guide,
            decode_init_state,
            self.lstm_decode,
            loop_function=None,
            scope="decoding"
        )
        # self.pred_line = pred_line
        print "pred line shape ", show_dim(pred_line)
        # apply a softmax to all the tokens in the pred line
        pred_line_softmax = [tf.nn.softmax(x) for x in pred_line]
        # add a small number to prevent blow up
        small_number = tf.constant(1e-10, shape=[batch_size, n_toks])
        pred_line_softmax = [x + small_number for x in pred_line_softmax]
        # self.pred_line_softmax = pred_line_softmax

        # the true output and cross-entropy
        lst_output = [self.data_out[:, j, :] for j in range(1, max_line_l)]
        xentropys = [lst_output[i] * tf.log(pred_line_softmax[i]) for i in range(max_line_l - 1)]
        # pack -> stack
        xentropy = -tf.reduce_sum(tf.stack(xentropys))
        self.xentropy = xentropy

        tvars = tf.trainable_variables()
        grads = tf.gradients(xentropy, tvars)
        grads = [tf.clip_by_value(grad, -2., 2.) for grad in grads]
        optimizer = tf.train.RMSPropOptimizer(0.002)
        # set the train_step so we can train it later
        self.train_step = optimizer.apply_gradients(zip(grads, tvars))

        # get a saver
        self.saver = tf.train.Saver()
        # restore model if we have a stored .ckpt file
        if model_loc != None:
            saver.restore(self.sess, model_loc)
            print "restored model"

    # ======= for generation with beam search =======
    def generate_usage_graph(self, model_loc):
        # I don't know why these are here but LOL!
        max_line_l = self.max_line_l
        n_toks = self.n_toks

        batch_size = self.batch_size
        encoder_units = self.encoder_units
        decoder_units = self.decoder_units
        dec_state_size = self.dec_state_size
        beam_batch_size = self.beam_batch_size
        beam_size = self.beam_size

        # set up the input for generation
        self.data_front = tf.placeholder(tf.float32, shape=(beam_batch_size, max_line_l, n_toks))
        self.data_front_l = tf.placeholder(tf.int32, shape=(beam_batch_size))
        self.data_back = tf.placeholder(tf.float32, shape=(beam_batch_size, max_line_l, n_toks))
        self.data_back_l = tf.placeholder(tf.int32, shape=(beam_batch_size))
        self.start_symbol = tf.placeholder(tf.float32, shape=(beam_batch_size, n_toks), name="start")
        self.stop_symbol = tf.placeholder(tf.int32, shape=(beam_batch_size, 1), name="stopp")

        # encoding
        # listify the front and back inputs
        lst_front = [self.data_front[:, j, :] for j in range(max_line_l)]
        lst_back = [self.data_back[:, j, :] for j in range(max_line_l)]

        # compress both front and back with lstm
        #    with tf.variable_scope("encoding") as scope:
        _, front_state = tf.nn.rnn(self.lstm_encode,
                                   lst_front,
                                   dtype=tf.float32,
                                   sequence_length=self.data_front_l,
                                   scope="enc_front")
        #    scope.reuse_variables()
        _, back_state = tf.nn.rnn(self.lstm_encode,
                                  lst_back,
                                  dtype=tf.float32,
                                  sequence_length=self.data_back_l,
                                  scope="enc_back")

        # concatenate the compressed front and back
        front_back = array_ops.concat(1, [front_state, back_state])

        # duct-tape the states to right dimensions
        decode_init_state = tf.matmul(front_back, self.duct_tape)

        # decoding with beam search
        state_gen = [[decode_init_state for i in range(beam_size)]]
        syms = [[self.start_symbol for i in range(beam_size)]]
        syms_book_keep = []
        parent_syms_book_keep = []
        #    log_probs = [[tf.constant(0.0, dtype=tf.float32, shape=[-1, 1]) for i in range(beam_size)]]
        log_probs = [[tf.constant(0.0, dtype=tf.float32, shape=[beam_batch_size, 1]) for i in range(beam_size)]]
        #
        self.log_prob_book_keep = []
        self.end_book_keep = []
        lstm_is_there = False
        with tf.variable_scope("decoding") as scope:
            # take 1 step forward
            for time in range(max_line_l):
                new_states = []
                joint_log_probs = []
                for i in range(beam_size):

                    # make sure we create the lstm unit in only the first pass, and re-use afterwards
                    if lstm_is_there:
                        scope.reuse_variables()
                    else:
                        lstm_is_there = True

                    input_old = syms[-1][i]
                    state_old = state_gen[-1][i]

                    #          print "old input shape ", show_dim(input_old)
                    #          print "old state shape ", show_dim(state_old)

                    output_new, state_new = self.lstm_decode(input_old, state_old)

                    # store the state
                    new_states.append(state_new)

                    # take the log probability of the output
                    output_log_prob = tf.log(tf.nn.softmax(output_new))
                    #          print "out logprob shape ", show_dim(output_log_prob)
                    parent_log_prob = log_probs[-1][i]
                    #          print "out parent shape ", show_dim(parent_log_prob)
                    parent_log_prob_tiled = tf.tile(parent_log_prob, [1, n_toks])
                    #          print "parent log prob tile shape ", show_dim(parent_log_prob_tiled)
                    joint_log_prob = parent_log_prob_tiled + output_log_prob
                    #          print "joint log prob shape ", show_dim(joint_log_prob)

                    # wipe out the first other stuff... as they create duplicates
                    # this is important or you won't get different seq at all
                    if time == 0 and i > 0:
                        joint_log_prob = tf.constant(-1e5, shape=[beam_batch_size, n_toks])
                    joint_log_probs.append(joint_log_prob)

                # set up the state for the next iteration
                state_gen.append(new_states)

                # join all the joint-probabilities together
                joint_probs = tf.concat(1, joint_log_probs)

                #        print " == ==joint probs shape ", show_dim(joint_probs)
                best_probs, indices = tf.nn.top_k(joint_probs, beam_size)

                #        print "best-prob shape ", show_dim(best_probs)
                #        print "indx shape ", show_dim(indices)

                # add in the new symbols
                symbols = tf.mod(indices, [n_toks for j in range(beam_size)])

                # book keeping
                #        print "symbol shape as index ", show_dim(symbols)
                syms_book_keep.append(symbols)

                # see if we generated end symbols
                # is_end = tf.equal(symbols, tf.constant(stop_sym, shape=[beam_batch_size, beam_size]))
                is_end = tf.equal(symbols, tf.tile(self.stop_symbol, [1, beam_size]))
                self.end_book_keep.append(is_end)

                # reset the probability to 0 if generate end
                #        print "best prob shape ", show_dim(best_probs)
                # do book keep here to have the raw prob of new-line symbol
                self.log_prob_book_keep.append(best_probs)
                reset_prob = tf.constant(-1e5, shape=(beam_batch_size, beam_size))
                best_probs = tf.select(is_end, reset_prob, best_probs)

                # do book keep here to have the zero-ed out prob at the new-line
                #        self.log_prob_book_keep.append(best_probs)
                symbols = tf.split(1, beam_size, symbols)
                symbols = [idx_to_sym(idx, n_toks, beam_batch_size) for idx in symbols]
                #        print "symbols shape as 1hot ", show_dim(symbols)

                # append the symbols
                syms.append(symbols)

                # add in the parent symbols
                parent_symbols = tf.floordiv(indices, [n_toks for j in range(beam_size)])

                # book keeping
                parent_syms_book_keep.append(parent_symbols)
                #        parent_symbols = tf.split(1, beam_size, parent_symbols)
                #        parent_symbols = [idx_to_sym(idx, n_toks, beam_batch_size) for idx in parent_symbols]
                #        parent_syms.append(parent_symbols)

                # apend the log probs
                best_probs = tf.split(1, beam_size, best_probs)
                log_probs.append(best_probs)

                print "%^&*(%^&*(%^&*( ", time

        self.syms_book_keep = syms_book_keep
        self.parent_syms_book_keep = parent_syms_book_keep

        # get a saver
        self.saver = tf.train.Saver()
        # restore model if we have a stored .ckpt file
        if model_loc != None:
            self.saver.restore(self.sess, model_loc)
            print "restored model"

    def make_data_dict(self, datz):
        ret = dict()
        _data_front, _data_front_l, _data_back, _data_back_l, _data_out = datz
        ret[self.data_front] = _data_front
        ret[self.data_front_l] = _data_front_l
        ret[self.data_back] = _data_back
        ret[self.data_back_l] = _data_back_l
        ret[self.data_out] = _data_out
        return ret

    def make_usage_data_dict(self, datz, start, stop):
        beam_batch_size = self.beam_batch_size
        ret = dict()
        _data_front, _data_front_l, _data_back, _data_back_l, _data_out = datz
        startt = np.array([start for i in range(beam_batch_size)])
        stopp = np.array([stop for i in range(beam_batch_size)])

        #    print "========== ALL SHAPES ==========="
        #    print "front ", show_dim(self.data_front), show_dim(_data_front)
        #    print _data_front
        #    print "front l ", show_dim(self.data_front_l), show_dim(_data_front_l)
        #    print _data_front_l
        #    print "startt ", show_dim(self.start_symbol), show_dim(startt)
        #    print startt
        #    print "stopp ", show_dim(self.stop_symbol), show_dim(stopp)
        #    print stopp

        ret[self.data_front] = _data_front
        ret[self.data_front_l] = _data_front_l
        ret[self.data_back] = _data_back
        ret[self.data_back_l] = _data_back_l
        ret[self.start_symbol] = startt
        ret[self.stop_symbol] = stopp
        return ret

    def init_vars(self):
        self.sess.run(tf.initialize_all_variables())

    def save_model(self, path):
        save_path = self.saver.save(self.sess, path)
        print("Model saved in file: %s" % save_path)

    def get_xentropy(self, datz):
        data_dict = self.make_data_dict(datz)
        outz = self.sess.run(self.xentropy, feed_dict=data_dict)
        return outz

    def train(self, datz):
        data_dict = self.make_data_dict(datz)
        self.sess.run(self.train_step, feed_dict=data_dict)

    def test(self):
        test_data_dict = self.get_data_dict(True)

    def generate_beam(self, datz, start, stop):
        data_dict = self.make_usage_data_dict(datz, start, stop)
        syms_bk = self.sess.run(self.syms_book_keep, feed_dict=data_dict)
        syms_par_bk = self.sess.run(self.parent_syms_book_keep, feed_dict=data_dict)
        log_prob_bk = self.sess.run(self.log_prob_book_keep, feed_dict=data_dict)
        return syms_bk, syms_par_bk, log_prob_bk


stat = {
    "bnd_seq_l": 100,  # ?
    "bnd_toks": 100  # ?
}
nn_config = {
    "batch_size": 64,
    "encoder_units": 50,
    "decoder_units": 50,
    "beam_batch_size": 10,
    "beam_size": 5
}
nn_beam = NN_SANDWICH_BEAM(stat, nn_config)
nn_beam.generate_train_graph()
nn_beam.train()
