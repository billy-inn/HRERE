from model import Model
import tensorflow as tf
import datetime
from utils import data_utils
import numpy as np
import config
from sklearn.metrics import average_precision_score

class BiLSTM(Model):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
            position_size, pretrained_embedding, wpe, hparams):
        super(BiLSTM, self).__init__(sequence_length, num_classes, vocab_size,
            embedding_size, position_size, pretrained_embedding, wpe, hparams)

        self.state_size = hparams.state_size
        self.hidden_layers = hparams.hidden_layers
        self.hidden_size = hparams.hidden_size
        self.wpe_size = hparams.wpe_size
        self.dense_keep_prob = hparams.dense_keep_prob
        self.rnn_keep_prob = hparams.rnn_keep_prob

        self.build()

    def add_placeholders(self):
        self.input_words = tf.placeholder(tf.int32, [None, config.BAG_SIZE, self.sequence_length],
                name="input_words")
        self.input_textlen = tf.placeholder(tf.int32, [None, config.BAG_SIZE], name="input_textlen")
        self.input_positions = tf.placeholder(tf.int32, [None, config.BAG_SIZE, 2,
            self.sequence_length], name="input_positions")
        self.input_labels = tf.placeholder(tf.int64, [None], name="input_labels")
        self.phase = tf.placeholder(tf.bool, name="phase")
        self.dense_dropout = tf.placeholder(tf.float32, name="dense_dropout")
        self.rnn_dropout = tf.placeholder(tf.float32, name="rnn_dropout")

        self.input_words_flatten = tf.reshape(self.input_words, [-1, self.sequence_length])
        self.input_textlen_flatten = tf.reshape(self.input_textlen, [-1])
        self.input_positions_flatten = tf.reshape(self.input_positions, [-1, 2,
            self.sequence_length])

    def create_feed_dict(self, input_words, input_textlen, input_positions,
            input_labels=None, phase=False, dense_dropout=1., rnn_dropout=1.):
        feed_dict = {
            self.input_words: input_words,
            self.input_textlen: input_textlen,
            self.input_positions: input_positions,
            self.phase: phase,
            self.dense_dropout: dense_dropout,
            self.rnn_dropout: rnn_dropout,
        }
        if input_labels is not None:
            feed_dict[self.input_labels] = input_labels
        return feed_dict

    def add_embedding(self):
        with tf.device('/cpu:0'), tf.name_scope("word_embedding"):
            W = tf.Variable(self.pretrained_embedding, trainable=False, dtype=tf.float32, name="W")
            self.embedded_words = tf.nn.embedding_lookup(W, self.input_words_flatten)

        with tf.device('/cpu:0'), tf.name_scope("position_embedding"):
            W = tf.Variable(self.wpe, trainable=False, dtype=tf.float32, name="W")
            self.wpe_chars = tf.nn.embedding_lookup(W, self.input_positions_flatten)

        self.input_sentences = tf.concat([self.embedded_words] +
                tf.unstack(self.wpe_chars, axis=1), 2)

    def add_hidden_layer(self, x, idx):
        dim = self.output_dim if idx == 0 else self.hidden_size
        with tf.variable_scope("hidden_%d" % idx):
            W = tf.get_variable("W", shape=[dim, self.hidden_size],
                    initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            b = tf.get_variable("b", shape=[self.hidden_size],
                    initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            h = tf.nn.xw_plus_b(x, W, b)
            h_norm = tf.layers.batch_normalization(h, training=self.phase)
            h_drop = tf.nn.dropout(tf.nn.relu(h_norm), self.dense_dropout, seed=config.RANDOM_SEED)
        return h_drop

    def add_prediction_op(self):
        self.add_embedding()

        with tf.name_scope("sentence_repr"):
            attention_w = tf.get_variable("attention_w", [self.state_size, 1])
            cell_forward = tf.contrib.rnn.LSTMCell(self.state_size)
            cell_backward = tf.contrib.rnn.LSTMCell(self.state_size)
            cell_forward = tf.contrib.rnn.DropoutWrapper(cell_forward,
                input_keep_prob=self.dense_dropout, output_keep_prob=self.rnn_dropout,
                seed=config.RANDOM_SEED)
            cell_backward = tf.contrib.rnn.DropoutWrapper(cell_backward,
                input_keep_prob=self.dense_dropout, output_keep_prob=self.rnn_dropout,
                seed=config.RANDOM_SEED)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_forward, cell_backward, self.input_sentences,
                sequence_length=self.input_textlen_flatten, dtype=tf.float32)
            outputs_added = tf.nn.tanh(tf.add(outputs[0], outputs[1]))
            alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(outputs_added,
                [-1, self.state_size]), attention_w), [-1, self.sequence_length]))
            alpha = tf.expand_dims(alpha, 1)
            self.sen_repr = tf.squeeze(tf.matmul(alpha, outputs_added))

        self.output_features = self.sen_repr
        self.output_dim = self.state_size

        with tf.name_scope("sentence_att"):
            attention_A = tf.get_variable("attention_A", shape=[self.output_dim])
            query_r = tf.get_variable("query_r", shape=[self.output_dim, 1])

            sen_repre = tf.tanh(self.output_features)
            sen_alpha = tf.expand_dims(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre,
                attention_A), query_r), [-1, config.BAG_SIZE])), 1)
            sen_s = tf.reshape(tf.matmul(sen_alpha, tf.reshape(sen_repre,
                [-1, config.BAG_SIZE, self.output_dim])), [-1, self.output_dim])

        h_drop = tf.nn.dropout(tf.nn.relu(sen_s), self.dense_dropout, seed=config.RANDOM_SEED)
        h_drop.set_shape([None, self.output_dim])
        h_output = tf.layers.batch_normalization(h_drop, training=self.phase)
        for i in range(self.hidden_layers):
            h_output = self.add_hidden_layer(h_output, i)

        with tf.variable_scope("output"):
            W = tf.get_variable("W", shape=[self.hidden_size, self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            b = tf.get_variable("b", shape=[self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            self.scores = tf.nn.xw_plus_b(h_output, W, b, name="scores")
            self.probs = tf.nn.softmax(self.scores, name="probs")
            self.predictions = tf.argmax(self.probs, 1, name="predictions")

    def add_loss_op(self):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_labels, logits=self.scores)
            self.l2_loss = tf.contrib.layers.apply_regularization(
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                weights_list=tf.trainable_variables())
            self.loss = tf.reduce_mean(losses) + self.l2_loss

        with tf.name_scope("accuracy"):
            mask = 1 - tf.to_float(tf.equal(self.input_labels, 0))
            correct_predictions = tf.to_float(tf.equal(self.predictions, self.input_labels))
            self.valid_size = tf.reduce_sum(mask)
            self.correct_num = tf.reduce_sum(mask * correct_predictions)
            # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.apply_gradients(self.grads_and_vars,
                global_step=self.global_step)

    def train_on_batch(self, sess, input_words, input_textlen, input_positions, input_labels):
        feed = self.create_feed_dict(input_words, input_textlen, input_positions,
                input_labels, True, self.dense_keep_prob, self.rnn_keep_prob)
        _, step, loss, size, cnt = sess.run(
            [self.train_op, self.global_step, self.loss, self.valid_size, self.correct_num],
            feed_dict=feed)
        acc = 0.0
        if size > 0:
            acc = cnt / size
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))

    def validation(self, sess, valid):
        batches = data_utils.batch_iter(valid, self.batch_size, 1, shuffle=False)
        total_loss = 0.0
        total_len = 0
        total_cnt = 0
        total_size = 0
        all_probs = np.zeros((0, self.num_classes - 1))
        all_labels = []
        for batch in batches:
            words_batch, textlen_batch, positions_batch, labels_batch = zip(*batch)
            feed = self.create_feed_dict(words_batch, textlen_batch, positions_batch, labels_batch)
            loss, size, cnt, probs = sess.run(
                [self.loss, self.valid_size, self.correct_num, self.probs],
                feed_dict=feed)
            total_loss += loss * len(labels_batch)
            total_len += len(labels_batch)
            total_cnt += cnt
            total_size += size
            all_probs = np.concatenate((all_probs, probs[:, 1:]))
            for l in labels_batch:
                tmp = np.zeros(self.num_classes - 1)
                if l > 0:
                    tmp[l - 1] = 1.0
                all_labels.append(tmp)
        all_probs = np.reshape(all_probs, (-1))
        all_labels = np.reshape(np.array(all_labels), (-1))
        average_precision = average_precision_score(all_labels, all_probs)
        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g} acc {:g} ap {:g}".format(time_str,
            total_loss / total_len, total_cnt / total_size, average_precision))
        return total_loss / total_len, total_cnt / total_size, average_precision

    def fit(self, sess, train, valid=None):
        train_batches = data_utils.batch_iter(train, self.batch_size, self.num_epochs)
        data_size = len(train)
        num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
        best_valid_acc = 0.0
        best_valid_loss = 1e10
        best_valid_ap = 0.0
        best_valid_epoch = 0
        for batch in train_batches:
            words_batch, textlen_batch, positions_batch, labels_batch = zip(*batch)
            self.train_on_batch(sess, words_batch, textlen_batch, positions_batch, labels_batch)
            current_step = tf.train.global_step(sess, self.global_step)
            if (current_step % num_batches_per_epoch == 0) and (valid is not None):
                print("\nEvaluation:")
                print("previous best valid epoch %d, best valid ap %.3f with loss %.3f acc %.3f" %
                    (best_valid_epoch, best_valid_ap, best_valid_loss, best_valid_acc))
                loss, acc, ap = self.validation(sess, valid)
                print("")
                if ap > best_valid_ap:
                    best_valid_loss = loss
                    best_valid_acc = acc
                    best_valid_ap = ap
                    best_valid_epoch = current_step // num_batches_per_epoch
                if current_step // num_batches_per_epoch - best_valid_epoch > 3:
                    break
        return best_valid_epoch, best_valid_loss, best_valid_acc, best_valid_ap

    def predict(self, sess, test):
        batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
        all_probs = np.zeros((0, self.num_classes - 1))
        all_labels = []
        total_cnt = 0
        total_size = 0
        for batch in batches:
            words_batch, textlen_batch, positions_batch, labels_batch = zip(*batch)
            feed = self.create_feed_dict(words_batch, textlen_batch, positions_batch, labels_batch)
            loss, probs, size, cnt = sess.run(
                [self.loss, self.probs, self.valid_size, self.correct_num],
                feed_dict=feed)
            total_cnt += cnt
            total_size += size
            all_probs = np.concatenate((all_probs, probs[:, 1:]))
            for l in labels_batch:
                tmp = np.zeros(self.num_classes - 1)
                if l > 0:
                    tmp[l - 1] = 1.0
                all_labels.append(tmp)
        all_probs = np.reshape(all_probs, (-1))
        all_labels = np.reshape(np.array(all_labels), (-1))
        return all_labels, all_probs, total_cnt / total_size

    def evaluate(self, sess, train, test):
        train_batches = data_utils.batch_iter(train, self.batch_size, self.num_epochs)
        data_size = len(train)
        num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
        for batch in train_batches:
            words_batch, textlen_batch, positions_batch, labels_batch = zip(*batch)
            self.train_on_batch(sess, words_batch, textlen_batch, positions_batch, labels_batch)
            current_step = tf.train.global_step(sess, self.global_step)
            if current_step % num_batches_per_epoch == 0:
                yield self.predict(sess, test)

    def save_preds(self, sess, test):
        batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
        all_labels = []
        all_preds = []
        for batch in batches:
            words_batch, textlen_batch, positions_batch, labels_batch = zip(*batch)
            feed = self.create_feed_dict(words_batch, textlen_batch, positions_batch, labels_batch)
            preds = sess.run(self.predictions, feed_dict=feed)
            all_labels = np.concatenate((all_labels, labels_batch))
            all_preds = np.concatenate((all_preds, preds))
        outfile = open("preds.txt", "w")
        for x, y in zip(all_preds, all_labels):
            if y == 0:
                continue
            outfile.write("%d %d\n" % (x, y))
        outfile.close()
