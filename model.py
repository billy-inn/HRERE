import tensorflow as tf

class Model(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
            position_size, pretrained_embedding, wpe, hparams):
        # data parameters
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.position_size = position_size
        self.pretrained_embedding = pretrained_embedding
        self.wpe = wpe

        # required params
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.lr = hparams.lr
        self.batch_size = hparams.batch_size
        self.num_epochs = hparams.num_epochs

        # global step for tensorflow
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

    def add_placeholders(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()
