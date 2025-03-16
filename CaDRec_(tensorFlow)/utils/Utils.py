# import torch
import Constants as C
# import torch.nn.functional as F
import tensorflow as tf
import tensorflow as tfn


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    return tf.cast(tf.not_equal(seq, C.PAD), tf.float32)[..., tf.newaxis]


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = tf.shape(seq_q)[1]
    ## Carl - Create mask
    padding_mask = tf.equal(seq_k, C.PAD)
    ## Carl - Add dimension for queries
    padding_mask = tf.expand_dims(padding_mask, axis = 1)  # b x lq x lk
    ## Carl - Tile the mask along the query dimension
    padding_mask = tf.tile(padding_mask, [1, len_q, 1])
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b = tf.shape(seq)[0]
    len_s = tf.shape(seq)[1]
    ## Carl - Create matrix of ones
    mask = tf.ones((len_s, len_s), dtype = tf.uint8)
    ## Carl - get the upper triangle part starting from the first diagonal above the main diagonal.
    subsequent_mask = tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0)
    ## Carl - Expand and tile for batch
    subsequent_mask = tf.expand_dims(subsequent_mask, axis = 0)
    subsequent_mask = tf.tile(subsequent_mask, [sz_b, 1, 1])
    return subsequent_mask


def type_loss(prediction, label, event_time, test_label, opt):
    """ Event prediction loss, cross entropy or label smoothing. """

    ## Carl - Squeeze prediction along axis 1
    #prediction = tf.squeeze(prediction[:, :], axis = 1)

    ## Carl - Create mutable tensor for multi_hots
    batch_size = tf.shape(label)[0]
    multi_hots = tf.Variable(tf.zeros([batch_size, C.ITEM_NUMBER], dtype = tf.float32))

    ## Carl - Convert tensors to numpy to iterate
    label_np = label.numpy() if hasattr(label, "numpy") else label
    test_label_np = test_label.numpy() if hasattr(test_label, "numpy") else test_label

    for i in range(batch_size):
        for val in label_np[i]:
            if val != 0:
                index = int(val) - 1
                multi_hots[i, index].assign(opt.beta)

        for val in test_label_np[i]:
            if val != 0:
                index = int(val) - 1
                multi_hots[i, index].assign(opt.lambda_)

    ## Carl - compute log probability
    log_prb = tf.math.log_sigmoid(prediction)
    ## Carl - apply label smoothing
    multi_hots = multi_hots * (1 - opt.smooth) + (1 - multi_hots) * opt.smooth / C.ITEM_NUMBER
    predict_loss = -(multi_hots * log_prb)

    loss = tf.reduce_sum(predict_loss)

    return loss


def l2_reg_loss(reg, model, event_type):
    """ Computes l2 regularization loss from event embeddings. """
    
    emb_loss = 0.0
    batch_size = tf.shape(event_type)[0]

    ## Carl - Convert event_type to numpy to iterate
    event_type_np = event_type.numpy() if hasattr(event_type, "numpy") else event_type

    for i in range(batch_size):
        e = event_type_np[i]
        ## Carl - Create mask for non-padding tokens
        mask = (e != C.PAD)
        e_masked = e[mask]

        if len(e_masked) > 0:
            ## Carl - Convert to tensor and get embeddings from model's event embedding layer
            r = model.event_emb(tf.convert_to_tensor(e_masked, dtype = tf.int32))
            emb_loss += tf.reduce_sum(tf.norm(r, ord = 2))

    return emb_loss / tf.cast(batch_size, tf.float32)


def popularity_loss(popularity_pred, in_degree):
    """ Computes Popularity Loss """
    return tf.square(popularity_pred - in_degree)

