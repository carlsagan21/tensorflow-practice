# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random

from six.moves import xrange
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
# from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder

import code_loader


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
      A loop function.
    """

    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
    """RNN decoder for the sequence-to-sequence model.

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor with shape [batch_size x cell.state_size].
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      loop_function: If not None, this function will be applied to the i-th output
        in order to generate the i+1-st input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
          (Note that in some cases, like basic RNN cell or GRU cell, outputs and
           states can be the same. They are different for LSTM cells though.)
    """
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None



        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)

            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
    """RNN decoder with embedding and a pure-decoding option.

    Args:
      decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
      initial_state: 2D Tensor [batch_size x cell.state_size].
      cell: core_rnn_cell.RNNCell defining the cell function.
      num_symbols: Integer, how many symbols come into the embedding.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_symbols] and B has
        shape [num_symbols]; if provided and feed_previous=True, each fed
        previous output will first be multiplied by W and added B.
      feed_previous: Boolean; if True, only the first of decoder_inputs will be
        used (the "GO" symbol), and all other decoder inputs will be generated by:
          next = embedding_lookup(embedding, argmax(previous_output)),
        In effect, this implements a greedy decoder. It can also be used
        during training to emulate http://arxiv.org/abs/1506.03099.
        If False, decoder_inputs are used as given (the standard decoder case).
      update_embedding_for_previous: Boolean; if False and feed_previous=True,
        only the embedding for the first symbol of decoder_inputs (the "GO"
        symbol) will be updated by back propagation. Embeddings for the symbols
        generated from the decoder itself remain unchanged. This parameter has
        no effect if feed_previous=False.
      scope: VariableScope for the created subgraph; defaults to
        "embedding_rnn_decoder".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors. The
          output is of shape [batch_size x cell.output_size] when
          output_projection is not None (and represents the dense representation
          of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
          when output_projection is None.
        state: The state of each decoder cell in each time-step. This is a list
          with length len(decoder_inputs) -- one item for each time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: When output_projection has the wrong shape.
    """
    with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
        if output_projection is not None:
            dtype = scope.dtype
            proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
            proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        embedding = variable_scope.get_variable("embedding",
                                                [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = (embedding_ops.embedding_lookup(embedding, i)
                   for i in decoder_inputs)
        return rnn_decoder(
            emb_inp, initial_state, cell, loop_function=loop_function)


def embedding_rnn_seq2seq(
        encoder_inputs,
        decoder_inputs,
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size,
        num_layers,
        output_projection=None,
        feed_previous=False,
        dtype=None,
        scope=None
):
    """Embedding RNN sequence-to-sequence model.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs RNN decoder, initialized with the last
    encoder state, on embedded decoder_inputs.

    Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_seq2seq"

    Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq", dtype=dtype) as scope:
        # if dtype is not None:
        #     scope.set_dtype(dtype)
        # else:
        dtype = scope.dtype
        # Encoder.
        encoder_cell = tf.contrib.rnn.EmbeddingWrapper(
            cell,
            embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size
        )

        encoder_inputs_front = []
        encoder_inputs_back = []
        for encoder_input in encoder_inputs:
            encoder_inputs_front.append(encoder_input[0])
            encoder_inputs_back.append(encoder_input[1])

        _, encoder_front_state = tf.contrib.rnn.static_rnn(
            encoder_cell, encoder_inputs_front, dtype=dtype, scope="enc_front"
        )

        _, encoder_back_state = tf.contrib.rnn.static_rnn(
            encoder_cell, encoder_inputs_back, dtype=dtype, scope="enc_back"
        )

        # 말이 되나? final state 만 쓰기
        front_back = array_ops.concat([encoder_front_state, encoder_back_state], 2)

        duct_tape = tf.Variable(
            initial_value=tf.random_normal(
                [num_layers, embedding_size * 2, embedding_size],
                name="tape"
            )
        )

        merged_encode_state = tf.matmul(front_back, duct_tape)
        decode_init_state = []
        for layer in xrange(merged_encode_state.get_shape()[0]):
            decode_init_state.append(merged_encode_state[layer])

        decode_init_state = tf.tuple(decode_init_state)

        # Decoder.
        # sampling 하면 정의되어 있음.
        if output_projection is None:
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_decoder_symbols)

        if isinstance(feed_previous, bool):
            return embedding_rnn_decoder(
                decoder_inputs,
                decode_init_state,
                cell,
                num_decoder_symbols,
                embedding_size,
                output_projection=output_projection,
                feed_previous=feed_previous
            )

            # If feed_previous is a Tensor, we construct 2 graphs and use cond.
            # def decoder(feed_previous_bool):
            #     reuse = None if feed_previous_bool else True
            #     with variable_scope.variable_scope(
            #             variable_scope.get_variable_scope(), reuse=reuse) as scope:
            #         outputs, state = embedding_attention_decoder(
            #             decoder_inputs,
            #             encoder_state,
            #             attention_states,
            #             cell,
            #             num_decoder_symbols,
            #             embedding_size,
            #             num_heads=num_heads,
            #             output_size=output_size,
            #             output_projection=output_projection,
            #             feed_previous=feed_previous_bool,
            #             update_embedding_for_previous=False,
            #             initial_state_attention=initial_state_attention)
            #         state_list = [state]
            #         if nest.is_sequence(state):
            #             state_list = nest.flatten(state)
            #         return outputs + state_list
            #
            # outputs_and_state = control_flow_ops.cond(feed_previous,
            #                                           lambda: decoder(True),
            #                                           lambda: decoder(False))
            # outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
            # state_list = outputs_and_state[outputs_len:]
            # state = state_list[0]
            # if nest.is_sequence(encoder_state):
            #     state = nest.pack_sequence_as(
            #         structure=encoder_state, flat_sequence=state_list)
            # return outputs_and_state[:outputs_len], state


class CodeModel(object):
    def __init__(
            self,
            vocab_size,
            buckets,
            size,
            num_layers,
            max_gradient_norm,
            batch_size,
            learning_rate,
            learning_rate_decay_factor,
            use_lstm=False,
            num_samples=512,
            forward_only=False,
            dtype=tf.float32
    ):

        self.vocab_size = vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.GRUCell(size)

        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.LSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        # TODO shape = [batch_size] or None?
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[2, None], name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return embedding_rnn_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=size,
                num_layers=num_layers,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype
            )

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function
            )
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function
            )

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                print("setting gradients and SGD update operation for bucket %d" % b)
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs_front, encoder_inputs_back, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs_front: list of numpy int vectors to feed as encoder inputs.
          encoder_inputs_back: list of numpy int vectors to feed as encoder inputs.
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
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs_front) != encoder_size:
            raise ValueError("Encoder front length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs_front), encoder_size))
        if len(encoder_inputs_back) != encoder_size:
            raise ValueError("Encoder back length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs_back), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = [encoder_inputs_front[l], encoder_inputs_back[l]]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id, random_set=True):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for idx in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id]) if random_set else data[bucket_id][idx]

            # Encoder inputs are padded and then reversed.
            encoder_pad_front = [code_loader.PAD_ID] * (encoder_size - len(encoder_input[0]))
            encoder_pad_back = [code_loader.PAD_ID] * (encoder_size - len(encoder_input[1]))
            encoder_inputs.append(
                [list(reversed(encoder_input[0] + encoder_pad_front)),
                 list(encoder_input[1] + encoder_pad_back)]
            )

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad = [code_loader.PAD_ID] * (decoder_size - len(decoder_input[0]) - 1)
            decoder_inputs.append([code_loader.GO_ID] + decoder_input[0] + decoder_pad)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs_front, batch_encoder_inputs_back, batch_decoder_inputs, batch_weights = [], [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs_front.append(
                np.array([encoder_inputs[batch_idx][0][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs_back.append(
                np.array([encoder_inputs[batch_idx][1][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == code_loader.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs_front, batch_encoder_inputs_back, batch_decoder_inputs, batch_weights
