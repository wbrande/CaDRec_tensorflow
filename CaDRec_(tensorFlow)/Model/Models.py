from utils import Utils

##import torch.nn
##import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout

from utils.Utils import *
from Model.hGCN import hGCNEncoder


# class Encoder(nn.Module):
#     def __init__(
#             self,
#             num_types, d_model, n_layers, n_head, dropout):
#         super().__init__()
#         self.d_model = d_model

#         self.layer_stack = nn.ModuleList([
#             hGCNEncoder(d_model, n_head)
#             for _ in range(n_layers)])

#     def forward(self, user_id, event_type, enc_output, user_output, adjacent_matrix):
#         """ Encode event sequences via masked self-attention. """

#         # get individual adj
#         adj = torch.zeros((event_type.size(0), event_type.size(1), event_type.size(1)), device='cuda:0')
#         for i, e in enumerate(event_type):
#             # the slicing operation
#             adj[i] = adjacent_matrix[e - 1, :][:, e - 1]
#             # performance can be enhanced by adding the element in the diagonal of the normalized adjacency matrix.
#             adj[i] += adjacent_matrix[e - 1, e - 1]

#         for i, enc_layer in enumerate(self.layer_stack):
#             residual = enc_output
#             enc_output = enc_layer(enc_output, user_output, adj, event_type)
#             if C.DATASET in {'douban-book'}:
#                 enc_output += residual

#         return enc_output.mean(1)


# class Predictor(nn.Module):
#     """ Prediction of next event type. """

#     def __init__(self, dim, num_types):
#         super().__init__()

#         self.dropout = nn.Dropout(0.5)
#         self.temperature = 512 ** 0.5
#         self.dim = dim

#     def forward(self, user_embeddings, embeddings, pop_encoding, evaluation):
#         outputs = []
#         if C.ABLATION != 'w/oMatcher':
#             if not evaluation:
#                 # C.BETA_1：
#                 #   Foursquare 0.5 Yelp2018 0.3 Gowalla 0.1 Brightkite 0.3 ml-1M 0.8 lastfm-2k 0.3 douban-book 0.05
#                 item_encoding = torch.concat([embeddings[1:], pop_encoding[1:] * C.BETA_1], dim=-1)
#                 out = user_embeddings.matmul(item_encoding.T)
#             else:
#                 item_encoding = embeddings[1:]
#                 out = user_embeddings[:, :self.dim].matmul(item_encoding.T)

#             # out = user_embeddings.matmul(embeddings.T[:,1:])
#             out = F.normalize(out, p=2, dim=-1, eps=1e-05)
#             outputs.append(out)

#         outputs = torch.stack(outputs, dim=0).sum(0)
#         out = torch.tanh(outputs)
#         return out


# class Model(nn.Module):
#     def __init__(
#             self, num_types, d_model=256, n_layers=4, n_head=4, dropout=0.1, device=0):
#         super(Model, self).__init__()

#         self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=C.PAD)
#         self.user_emb = nn.Embedding(C.USER_NUMBER, d_model, padding_idx=C.PAD)

#         self.encoder = Encoder(
#             num_types=num_types, d_model=d_model,
#             n_layers=n_layers, n_head=n_head, dropout=dropout)
#         self.num_types = num_types

#         self.predictor = Predictor(d_model, num_types)

#     def forward(self, user_id, event_type, adjacent_matrix, pop_encoding, evaluation=True):

#         non_pad_mask = Utils.get_non_pad_mask(event_type)

#         # (K M)  event_emb: Embedding
#         enc_output = self.event_emb(event_type)
#         user_output = self.user_emb(user_id)

#         pop_output = pop_encoding[event_type] * non_pad_mask

#         if C.ABLATION != 'w/oUSpec' and C.ABLATION != 'w/oDisen':
#             enc_output += torch.sign(enc_output)\
#                           * F.normalize(user_output.unsqueeze(1), dim=-1) # * torch.sign(user_output.unsqueeze(1)) \

#         output = self.encoder(user_id, event_type, enc_output, user_output, adjacent_matrix)

#         user_embeddings = torch.concat([output, torch.mean(pop_output, dim=1) * C.BETA_1], dim=-1)

#         prediction = self.predictor(user_embeddings, self.event_emb.weight, pop_encoding, evaluation)

#         return prediction, user_embeddings, pop_output


## Carl - New class Encoder for implementing with TensorFlow
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_types, d_model, n_layers, n_head, dropout):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.layer_stack = [hGCNEncoder(d_model, n_head) for _ in range(n_layers)]
        self.dropout = Dropout(dropout)

    def call(self, user_id, event_type, enc_output, user_output, adjacent_matrix, training=False):
        batch_size = tf.shape(user_id)[0]
        seq_len = tf.shape(event_type)[1]
    
    # If the sequence length is zero, return a tensor of zeros with an appropriate shape.
        if seq_len == 0:
        # For example, returning a zero vector for each sample.
            return tf.zeros((batch_size, self.d_model))
    
    # Initialize the tensor that will hold the per-sample adjacency matrices.
        adj = tf.zeros((batch_size, seq_len, seq_len), dtype=tf.float32)
    
        for i in range(batch_size):
        # Extract the i-th sample's event sequence.
            e = event_type[i]  # shape: [seq_len]
        # Check if there are any valid events (non-padding)
            if e.shape[0] == 0 or tf.reduce_sum(tf.cast(e > 0, tf.int32)) == 0:
                continue  # Skip updating this sample if it's empty.
        
        # Create indices safely: replace padded zeros with 0 so that e - 1 doesn't produce -1.
            indices = tf.where(e > 0, e - 1, tf.zeros_like(e))
        
        # Gather rows from the adjacent matrix using these indices.
            rows = tf.gather(adjacent_matrix, indices)
        # Then gather columns using the same indices along axis=1.
            adj_i = tf.gather(rows, indices, axis=1)
        
        # Update the overall tensor. Make sure to expand dims so that update shape is [1, seq_len, seq_len].
            adj = tf.tensor_scatter_nd_update(adj, [[i]], tf.expand_dims(adj_i, axis=0))
    
    # Continue with the rest of the Encoder call logic...
    # For example, apply subsequent operations on enc_output and so on.
    # Here, we simply return an average as a placeholder:
        return tf.reduce_mean(enc_output, axis=1)

    def call_old(self, user_id, event_type, enc_output, user_output, adjacent_matrix, training = False):
        """ Encode event sequences via masked self-attention. """

        ## Carl - get individual adjacency matrix for the different event sequences
        batch_size = tf.shape(event_type)[0]
        seq_len = tf.shape(event_type)[1]
        adj = tf.zeros((batch_size, seq_len, seq_len), dtype = tf.float32)

        for i in range(batch_size):
            e = event_type[i]

            indices = tf.where(e > 0, e - 1, tf.zeros_like(e))
            rows = tf.gather(adjacent_matrix, indices)
            adj_i = tf.gather(rows, indices, axis=1)

            adj = tf.tensor_scatter_nd_update(adj, [[i]], tf.expand_dims(adj_i, axis=0))

        for i in range(batch_size):
            e = event_type[i]
            # Compute valid indices: replace padded zeros with a safe value and get valid ones.
            valid_mask = tf.greater(e, 0)
            if tf.reduce_sum(tf.cast(valid_mask, tf.int32)).numpy() == 0:
            # If there are no valid indices, skip the update for this sample.
                continue
            indices = tf.where(valid_mask, e - 1, tf.zeros_like(e))
            # Gather rows and then columns from adjacent_matrix.
            rows = tf.gather(adjacent_matrix, indices)
            adj_i = tf.gather(rows, indices, axis=1)
            # Now update: ensure update tensor has the correct extra dimension.
            adj = tf.tensor_scatter_nd_update(adj, [[i]], tf.expand_dims(adj_i, axis=0))

        ## Carl - apply stacked hGCN layers
        for enc_layer in self.layer_stack:
            residual = enc_output
            enc_output = enc_layer(enc_output, user_output, adj, event_type)
            if C.DATASET in {'douban-book'}:
                enc_output += residual

        ## Carl - replaced return for TensorFlow
        return tf.reduce_mean(enc_output, axis = 1)

        # return enc_output.mean(1)


## Carl - New class Predictor for implementing with TensorFlow
class Predictor(tf.keras.layers.Layer):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super(Predictor, self).__init__()

        self.dropout = Dropout(0.5)
        self.temperature = tf.sqrt(512.0)
        self.dim = dim

    def call(self, user_embeddings, embeddings, pop_encoding, evaluation, training = False):
        outputs = []

        if C.ABLATION != 'w/oMatcher':
            if not evaluation:
                # C.BETA_1：
                #   Foursquare 0.5 Yelp2018 0.3 Gowalla 0.1 Brightkite 0.3 ml-1M 0.8 lastfm-2k 0.3 douban-book 0.05
                item_encoding = tf.concat([embeddings[1:], pop_encoding[1:] * C.BETA_1], axis =- 1)
                out = tf.matmul(user_embeddings, tf.transpose(item_encoding))
            else:
                item_encoding = embeddings[1:]
                out = tf.matmul(user_embeddings[:, :self.dim], tf.transpose(item_encoding))

            ## Carl - Output Normalization
            out = tf.nn.l2_normalize(out, axis =- 1, epsilon = 1e-05)
            outputs.append(out)

        outputs = tf.reduce_sum(tf.stack(outputs, axis = 0), axis = 0)
        return tf.tanh(outputs)


## Carl - New class model for implementing with TensorFlow
class Model(tf.keras.Model):
    def __init__(self, num_types, d_model=256, n_layers=4, n_head=4, dropout=0.1):
        super(Model, self).__init__()

        self.event_emb = Embedding(num_types+1, d_model, mask_zero=True)
        self.user_emb = Embedding(C.USER_NUMBER, d_model, mask_zero=True)

        self.num_types = num_types
        self.encoder = Encoder(num_types, d_model, n_layers, n_head, dropout)
        self.predictor = Predictor(d_model, num_types)

    def call(self, user_id, event_type, adjacent_matrix, pop_encoding, evaluation, training=False):
        non_pad_mask = Utils.get_non_pad_mask(event_type)

        enc_output = self.event_emb(event_type)
        user_output = self.user_emb(user_id)

        pop_output = tf.gather(pop_encoding, event_type) * non_pad_mask


        if C.ABLATION not in {'w/oUSpec', 'w/oDisen'}:
            enc_output += tf.sign(enc_output) * tf.nn.l2_normalize(tf.expand_dims(user_output, 1), axis =- 1)

        output = self.encoder(user_id, event_type, enc_output, user_output, adjacent_matrix)

        user_embeddings = tf.concat([output, tf.reduce_mean(pop_output, axis = 1) * C.BETA_1], axis =- 1)
        prediction = self.predictor(user_embeddings, self.event_emb.weights[0], pop_encoding, evaluation=evaluation)
        
        return prediction, user_embeddings, pop_output
