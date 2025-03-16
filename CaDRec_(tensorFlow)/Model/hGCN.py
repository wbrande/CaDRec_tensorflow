# import torch
from utils import Utils
# import torch.nn as nn
import Constants as C
# import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow.nn as tfn

## Carl - Stacks multiple hGCNLayers and collects their outputs
class hGCNEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head):
        super(hGCNEncoder, self).__init__()

        ## Carl - Create list of hGCNLayers for each attention head
        self.heads = [hGCNLayer(d_model, d_model) for _ in range(n_head)]

    def get_non_pad_mask(self, seq):
        """ Get the non-padding positions. """
        return Utils.get_non_pad_mask(seq)

    def call(self, output, user_output, sparse_norm_adj, event_type, training = False):
        ## Carl - Computer subsequent mask and key padding mask using tensorflow
        slf_attn_mask_subseq = Utils.get_subsequent_mask(event_type)  # M * L * L
        slf_attn_mask_keypad = Utils.get_attn_key_pad_mask(event_type, event_type)  # M x lq x lk
        slf_attn_mask_keypad = tf.cast(slf_attn_mask_keypad, dtype = tf.float32)
        slf_attn_mask_subseq = tf.cast(slf_attn_mask_subseq, dtype = tf.float32)

        # Carl - Combine masks
        slf_attn_mask = tf.cast(tf.greater(slf_attn_mask_keypad + slf_attn_mask_subseq, 0), tf.float32)

        outputs = []
        ## Carl - Multiply output by non-pad mask
        output = output * self.get_non_pad_mask(event_type)
        ## Carl - Loop through eac attention head and collect results
        for head in self.heads:
            head_output = head(output, user_output, sparse_norm_adj, event_type, slf_attn_mask, training = training)
            outputs.append(output)
        ## Carl - Stack outputs along a different axis and then sum that axis
        outputs_stack = tf.stack(outputs, axis = 0)

        return tf.reduce_sum(outputs_stack, axis = 0)

## Carl - Basic layer used by the encoder
class hGCNLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_k):
        super(hGCNLayer, self).__init__()

        self.linear = layers.Dense(d_model, kernel_initializer = initializers.GlorotUniform())

        ## Carl - Define linear projections for query and key w/o bias
        self.w_qs = layers.Dense(d_k, use_bias=False, kernel_initializer = initializers.GlorotUniform())
        self.w_ks = layers.Dense(d_k, use_bias=False, kernel_initializer = initializers.GlorotUniform())

        self.temperature = tf.cast(tf.sqrt(tf.cast(d_model, tf.float32)), tf.float32)
        self.dropout = layers.Dropout(0.1)

    def call(self, output, user_output, sparse_norm_adj, event_type, slf_attn_mask, training = False):

        ## Carl - get queries and keys
        q = self.w_qs(output)
        k = self.w_ks(output)

        scaled_q = q / self.temperature

        attn = tf.matmul(scaled_q, tf.transpose(k, perm = [0, 2, 1]))
        ## Carl - apply self attention matrix
        attn = attn * tf.cast(slf_attn_mask, tf.float32)
        ## Apply dropout if training
        attn = self.dropout(attn, training = training)

        ## Carl - set epsilon based on dataset
        if C.DATASET == 'Foursquare':
            eps = 0.1
        elif C.DATASET == 'lastfm-2k':
            eps = 0.1
        elif C.DATASET == 'douban-book':
            eps = 0.1
        elif C.DATASET == 'Yelp2018':
            eps = 0.5
        else:
            eps = 0.0

        ## Carl - Conditional for ablation studies
        if C.ABLATION == 'OlyAtten':
            output = tf.matmul(attn, tfn.elu(self.linear(output)))
        elif C.ABLATION == 'OlyHGCN' or C.ABLATION == 'w/oSA':
            output = tf.matmul(sparse_norm_adj, tfn.elu(self.linear(output)))
        elif C.ABLATION == 'Addition':
            output = tf.matmul(sparse_norm_adj + attn, tfn.elu(self.linear(output)))
        else:
            ## Carl - modify aggregation based on dataset, if no ablation studies are occuring
            if C.DATASET == 'Foursquare':
                modified_attn = tf.sign(sparse_norm_adj) * tf.sign(attn) * tf.nn.l2_normalize(attn, axis =- 1) * eps
                output = tf.matmul(sparse_norm_adj + modified_attn, tfn.elu(self.linear(output)))
            else:
                modified_attn = tf.sign(sparse_norm_adj) * tf.nn.l2_normalize(attn, axis =- 1) * eps
                output = tf.matmul(sparse_norm_adj + modified_attn, tfn.elu(self.linear(output)))
                
        return output



