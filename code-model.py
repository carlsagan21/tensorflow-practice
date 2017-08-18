import numpy as np
import tensorflow as tf


class CodeModel:
    def __init__(
            self,
            # source_vocab_size,
            # target_vocab_size,
            # buckets,
            size,
            num_layers,
            # max_gradient_norm,
            batch_size,
            learning_rate,
            # learning_rate_decay_factor,
            use_lstm=False,
            # num_samples=512,
            # forward_only=False,
            dtype=tf.float32
    ):
        """Create the model.

            Args:
              source_vocab_size: size of the source vocabulary.
              target_vocab_size: size of the target vocabulary.
              buckets: a list of pairs (I, O), where I specifies maximum input length
                that will be processed in that bucket, and O specifies maximum output
                length. Training instances that have inputs longer than I or outputs
                longer than O will be pushed to the next bucket and padded accordingly.
                We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
              size: number of units in each layer of the model.
              num_layers: number of layers in the model.
              max_gradient_norm: gradients will be clipped to maximally this norm.
              batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
              learning_rate: learning rate to start with.
              learning_rate_decay_factor: decay learning rate by this much when needed.
              use_lstm: if true, we use LSTM cells instead of GRU cells.
              num_samples: number of samples for sampled softmax.
              forward_only: if set, we do not construct the backward pass in the model.
              dtype: the data type to use to store internal variables.
            """

        # self.source_vocab_size = source_vocab_size
        # self.target_vocab_size = target_vocab_size
        # self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype
        )
        # self.learning_rate_decay_op = self.learning_rate.assign(
        #     self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            if use_lstm:
                return tf.contrib.rnn.LSTMCell(
                    num_units=size,
                    use_peepholes=False,
                    cell_clip=None,
                    initializer=None,
                    num_proj=None,
                    proj_clip=None,
                    num_unit_shards=None,
                    num_proj_shards=None,
                    forget_bias=1.0,
                    state_is_tuple=True,
                    activation=None,
                    reuse=None
                )
            else:
                return tf.contrib.rnn.GRUCell(
                    num_units=size,
                    activation=None,
                    reuse=None,
                    kernel_initializer=None,
                    bias_initializer=None
                )

        def multi_cell():
            return tf.contrib.rnn.MultiRNNCell(
                cells=[single_cell() for _ in range(num_layers)],
                state_is_tuple=True
            )

        # cell = single_cell()
        # if num_layers > 1:
        #     cell = multi_cell()

        # encoder
        self.encoder = multi_cell()

        # decoder
        self.decoder = multi_cell()

    def step(
            self,
            session,
            encoder_inputs,
            decoder_inputs,
            target_weights,
            # bucket_id,
            # forward_only
    ):
        """Run a step of the model feeding the given inputs.

            Args:
              session: tensorflow session to use.
              encoder_inputs: list of numpy int vectors to feed as encoder inputs.
              decoder_inputs: list of numpy int vectors to feed as decoder inputs.
              target_weights: list of numpy float vectors to feed as target weights.
              bucket_id: which bucket of the model to use.
              forward_only: whether to do the backward step or only forward.

            Returns:
              A triple consisting of gradient norm (or None if we did not do backward),
              average perplexity, and the outputs.

            Raises:
              ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
            """

        # Check if the sizes match.
        # encoder_size, decoder_size = self.buckets[bucket_id]
        # if len(encoder_inputs) != encoder_size:
        #     raise ValueError("Encoder length must be equal to the one in bucket,"
        #                      " %d != %d." % (len(encoder_inputs), encoder_size))
        # if len(decoder_inputs) != decoder_size:
        #     raise ValueError("Decoder length must be equal to the one in bucket,"
        #                      " %d != %d." % (len(decoder_inputs), decoder_size))
        # if len(target_weights) != decoder_size:
        #     raise ValueError("Weights length must be equal to the one in bucket,"
        #                      " %d != %d." % (len(target_weights), decoder_size))

        # encoder_size = len(encoder_inputs)
        # decoder_size = len(decoder_inputs)
        #
        # input_feed = {}
        # for l in xrange(encoder_size):
        #     input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        # for l in xrange(decoder_size):
        #     input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        #     input_feed[self.target_weights[l].name] = target_weights[l]
        #
        # last_target = self.decoder_inputs[decoder_size].name
        # input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        _, front_state = tf.nn.dynamic_rnn(
            cell=self.encoder,
            inputs,
            sequence_length=None,
            initial_state=None,
            dtype=None,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope=None
        )