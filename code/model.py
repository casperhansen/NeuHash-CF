import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import numpy as np
from tensorflow.losses import compute_weighted_loss, Reduction

def hinge_loss_eps(labels, logits, epsval, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.to_float(logits)
    labels = math_ops.to_float(labels)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_eps = array_ops.ones_like(labels)*epsval
    all_ones = array_ops.ones_like(labels)

    labels = math_ops.subtract(2 * labels, all_ones)
    losses = nn_ops.relu(
        math_ops.subtract(all_eps, math_ops.multiply(labels, logits)))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)

class Model():
    def __init__(self, sample, args):
        self.sample = sample
        self.batchsize = args["batchsize"]

    def _make_embedding(self, vocab_size, embedding_size, name, trainable=True):
        W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size], minval=-1, maxval=1),
                        trainable=trainable, name=name)

        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        embedding_init = W.assign(embedding_placeholder)
        return (W, embedding_placeholder, embedding_init)

    def convert_sparse_matrix_to_sparse_tensor(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        res = tf.SparseTensor(indices, coo.data, coo.shape) # SparseTensorValue
        return res

    def _extract(self, item_emb, user_emb, content_matrix, batchsize):#, user_index_embedding_matrix, user_index_embedding_ratings_matrix):
        user, i1, i2, iu, i1r, i2r = self.sample[0], self.sample[1], self.sample[2], self.sample[3], self.sample[4], self.sample[5]

        #user_mask = tf.nn.embedding_lookup(user_mask_emb, user)

        #user_index_values = tf.cast(tf.nn.embedding_lookup(user_index_embedding_matrix, user), tf.int32)
        #user_index_ratings_values = tf.nn.embedding_lookup(user_index_embedding_ratings_matrix, user)

        #idx = tf.squeeze(tf.where(tf.not_equal(i1, -1)),-1)
        #sparse = tf.SparseTensor(idx, i1, [batchsize,]) # tf.gather_nd(i1, idx)
        #print(idx, sparse)
        #exit()
        content_feature_vector = tf.nn.embedding_lookup(content_matrix, i1)
        #print(content_feature_vector)
        #exit()
        user = tf.nn.embedding_lookup(user_emb, user) #tf.concat([tf.expand_dims(tf.nn.embedding_lookup(user_emb_x, user), axis=-1) for user_emb_x in user_embs], axis=-1)

        i1 = tf.nn.embedding_lookup(item_emb, i1)


        return user, i1, i1r, content_feature_vector # i2, i3, i1r, i2r#, user_index_values, user_index_ratings_values

    def _sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self._sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature, axis=-1)

    def gumbel_softmax(self, logits, temperature, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, bits, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, bits, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self._gumbel_softmax_sample(logits, temperature)
        #if hard:

        # k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keepdims=True)), y.dtype)
        #print(tf.reduce_max(y, -1), y)
        #exit()
        y_hard = tf.stop_gradient(y_hard - y) + y

        y = tf.cond(hard, lambda: y_hard, lambda: y)
        return y

    def make_importance_embedding(self, vocab_size, trainable=True):
        W = tf.Variable(tf.random_uniform(shape=[vocab_size], minval=0.1, maxval=1),
                        trainable=trainable, name="importance_embedding")
        return W

    def make_network(self, word_emb_matrix, importance_emb_matrix, content_matrix, item_emb, user_emb, is_training, args, max_rating, sigma_anneal, sigma_anneal_vae, batchsize):
        # user_index_embedding_matrix, user_index_embedding_ratings_matrix,

        #################### Bernoulli Sample #####################
        ## ref code: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        def bernoulliSample(x):
            """
            Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
            using the straight through estimator for the gradient.
            E.g.,:
            if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
            and the gradient will be pass-through (identity).
            """
            g = tf.get_default_graph()

            with ops.name_scope("BernoulliSample") as name:
                with g.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):

                    if args["deterministic_train"]:
                        train_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.ones(tf.shape(x)) * 0.5)
                    else:
                        train_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.random_uniform(tf.shape(x)))

                    if args["deterministic_eval"]:
                        eval_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.ones(tf.shape(x)) * 0.5)
                    else:
                        eval_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.random_uniform(tf.shape(x)))

                    mus = tf.cond(is_training, train_fn, eval_fn)

                    return tf.ceil(x - mus, name=name)

        @ops.RegisterGradient("BernoulliSample_ST")
        def bernoulliSample_ST(op, grad):
            return [grad, tf.zeros(tf.shape(op.inputs[1]))]

        ###########################################################
        def encoder(doc, hidden_neurons_encode, encoder_layers):
            doc_layer = tf.layers.dense(doc, hidden_neurons_encode, name="encode_layer0",
                                        reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            # doc_layer = tf.nn.dropout(doc_layer, dropout_keep)

            for i in range(1, encoder_layers):
                doc_layer = tf.layers.dense(doc_layer, hidden_neurons_encode, name="encode_layer" + str(i),
                                            reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

            doc_layer = tf.nn.dropout(doc_layer, tf.cond(is_training, lambda: 0.8, lambda: 1.0))

            doc_layer = tf.layers.dense(doc_layer,  args["bits"], name="last_encode", reuse=tf.AUTO_REUSE,
                                        activation=tf.nn.sigmoid)

            bit_vector = bernoulliSample(doc_layer)

            return bit_vector, doc_layer

        user, i1, i1r, i1_content = self._extract(item_emb, user_emb, content_matrix, batchsize)

        i1_content = i1_content * importance_emb_matrix
        content_hashcode, content_cont = encoder(i1_content, args["vae_units"], args["vae_layers"])
        e = tf.random.normal([batchsize, args["bits"]])
        noisy_content_hashcode = tf.math.multiply(e, sigma_anneal_vae) + content_hashcode

        softmax_bias = tf.Variable(tf.zeros(8000), name="softmax_bias")
        embedding = tf.layers.dense(word_emb_matrix, args["bits"], name="lower_dim_embedding_layer")
        #print(tf.multiply(tf.transpose(embedding), importance_emb_matrix))
        #print(embedding, importance_emb_matrix, noisy_content_hashcode)
        #exit()
        dot_emb_vector = tf.linalg.matmul(noisy_content_hashcode, #tf.transpose(embedding)) + softmax_bias
                                          tf.multiply(tf.transpose(embedding), importance_emb_matrix)) + softmax_bias

        softmaxed = tf.nn.softmax(dot_emb_vector)
        logaritmed = tf.math.log(tf.maximum(softmaxed, 1e-10))
        logaritmed = tf.multiply(logaritmed, tf.cast(i1_content > 0, tf.float32))
        loss_recon = tf.reduce_sum(tf.reduce_sum(logaritmed, 1), axis=0)

        loss_kl = tf.multiply(content_cont, tf.math.log(tf.maximum(content_cont / 0.5, 1e-10))) + \
                  tf.multiply(1 - content_cont, tf.math.log(tf.maximum((1 - content_cont) / 0.5, 1e-10)))
        loss_kl = tf.reduce_sum(tf.reduce_sum(loss_kl, 1), axis=0)
        loss_vae = -(loss_recon)# - args["KLweight"] * loss_kl)

        user = tf.sigmoid(user)
        user_sampling = user
        i1 = tf.sigmoid(i1)
        i1_sampling = i1

        user = bernoulliSample(user)
        if args["item_emb_type"] == 0:
            i1_org = bernoulliSample(i1)
        elif args["item_emb_type"] == 1:
            i1_org = content_hashcode
        elif args["item_emb_type"] == 2:
            i1_org = bernoulliSample(0.5*(i1+content_cont))
        else:
            exit(-1)

        i1_org_noselfmask = i1_org
        if args["optimize_selfmask"]:
            i1_org = i1_org * user

        user_m = 2*user - 1
        i1_org_m = (2*i1_org - 1)
        i1_org_m_noselfmask = (2*i1_org_noselfmask - 1)

        nonzero_bits = args["bits"]

        def make_total_loss(i1_org, i1r, i1_sampling, anneal):
            e0 = tf.random.normal([batchsize], stddev=1.0, name='normaldis0')
            i1 = i1_org
            i1r = i1r + e0*anneal

            i1r_m = 2*nonzero_bits * (i1r/max_rating) - nonzero_bits

            dot_i1 = tf.reduce_sum(user_m * i1, axis=-1)


            if args["loss_type"] == "normal":
                print("### normal")
                sqr_diff = tf.math.pow(i1r_m - dot_i1, 2)
                loss = tf.reduce_mean(sqr_diff, axis=-1)
            elif args["loss_type"] == "power":
                print("### power")
                pred_isLess = tf.cast(dot_i1 <= i1r_m, tf.float32)
                pred_isGreater = tf.cast(dot_i1 > i1r_m, tf.float32)

                less_part = tf.math.pow(i1r_m - dot_i1, 2)
                greater_part = tf.math.pow(i1r_m - dot_i1, args["loss_alpha"]) # alpha > 2

                loss = pred_isLess*less_part + pred_isGreater*greater_part
                loss = tf.reduce_mean(loss, axis=-1)

            elif args["loss_type"] == "mult":
                print("### mult")
                pred_isLess = tf.cast(dot_i1 <= i1r_m, tf.float32)
                pred_isGreater = tf.cast(dot_i1 > i1r_m, tf.float32)

                less_part = tf.math.pow(i1r_m - dot_i1, 2)
                greater_part = tf.math.pow(i1r_m - dot_i1, 2)*args["loss_alpha"] # alpha > 1

                loss = pred_isLess * less_part + pred_isGreater * greater_part
                loss = tf.reduce_mean(loss, axis=-1)
            else:
                exit(-1)
            # reg
            loss_kl = tf.multiply(i1_sampling, tf.math.log(tf.maximum(i1_sampling / 0.5, 1e-10))) + \
                      tf.multiply(1 - i1_sampling, tf.math.log(tf.maximum((1 - i1_sampling) / 0.5, 1e-10)))
            loss_kl = tf.reduce_sum(tf.reduce_sum(loss_kl, 1), axis=0)

            loss_kl_user = tf.multiply(user_sampling, tf.math.log(tf.maximum(user_sampling / 0.5, 1e-10))) + \
                      tf.multiply(1 - user_sampling, tf.math.log(tf.maximum((1 - user_sampling) / 0.5, 1e-10)))
            loss_kl_user = tf.reduce_sum(tf.reduce_sum(loss_kl_user, 1), axis=0)

            # combine losses
            total_loss = loss + args["KLweight"]*(loss_kl + loss_kl_user)

            i1_dist = -dot_i1
            return total_loss, i1_dist, loss

        total_loss, ham_dist_i1, reconloss = make_total_loss(i1_org_m, i1r, i1_sampling, sigma_anneal)

        if args["item_emb_type"] == 0:
            total_loss = total_loss
        else:
            total_loss = total_loss + args["vae_weight"] * loss_vae

        return total_loss, reconloss, ham_dist_i1, i1_org_m_noselfmask, user_m, reconloss, loss_vae
