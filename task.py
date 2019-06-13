from model_param_space import ModelParamSpace
from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval
from optparse import OptionParser
from utils import logging_utils, embedding_utils, pkl_utils
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import average_precision_score
import os
import config
import datetime
import tensorflow as tf
from bilstm import BiLSTM
from complex_hrere import ComplexHRERE
from real_hrere import RealHRERE

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Task:
    def __init__(self, model_name, runs, params_dict, logger):
        print("Loading data...")
        words, positions, heads, tails, labels = pkl_utils._load(config.GROUPED_TRAIN_DATA)
        words_test, positions_test, heads_test, tails_test, labels_test = pkl_utils._load(config.GROUPED_TEST_DATA) # noqa

        self.embedding = embedding_utils.Embedding(
            config.EMBEDDING_DATA,
            list([s for bags in words for s in bags]) +
            list([s for bags in words_test for s in bags]),
            config.MAX_DOCUMENT_LENGTH)

        print("Preprocessing data...")
        textlen = np.array([[self.embedding.len_transform(x) for x in y] for y in words])
        words = np.array([[self.embedding.text_transform(x) for x in y] for y in words])
        positions = np.array([[self.embedding.position_transform(x) for x in y] for y in positions])

        textlen_test = np.array([[self.embedding.len_transform(x) for x in y] for y in words_test])
        words_test = np.array([[self.embedding.text_transform(x) for x in y] for y in words_test])
        positions_test = np.array([[self.embedding.position_transform(x) for x in y] for y in positions_test]) # noqa

        ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=config.RANDOM_SEED)
        for train_index, valid_index in ss.split(np.zeros(len(labels)), labels):
            words_train, words_valid = words[train_index], words[valid_index]
            textlen_train, textlen_valid = textlen[train_index], textlen[valid_index]
            positions_train, positions_valid = positions[train_index], positions[valid_index]
            heads_train, heads_valid = heads[train_index], heads[valid_index]
            tails_train, tails_valid = tails[train_index], tails[valid_index]
            labels_train, labels_valid = labels[train_index], labels[valid_index]
        if "hrere" in model_name:
            self.full_set = list(zip(words, textlen, positions, heads, tails, labels))
            self.train_set = list(zip(words_train, textlen_train, positions_train, heads_train, tails_train, labels_train)) # noqa
            self.valid_set = list(zip(words_valid, textlen_valid, positions_valid, heads_valid, tails_valid, labels_valid)) # noqa
            self.test_set = list(zip(words_test, textlen_test, positions_test, heads_test, tails_test, labels_test)) # noqa
            if "complex" in model_name:
                self.entity_embedding1 = np.load(config.ENTITY_EMBEDDING1)
                self.entity_embedding2 = np.load(config.ENTITY_EMBEDDING2)
                self.relation_embedding1 = np.load(config.RELATION_EMBEDDING1)
                self.relation_embedding2 = np.load(config.RELATION_EMBEDDING2)
            else:
                self.entity_embedding = np.load(config.ENTITY_EMBEDDING)
                self.relation_embedding = np.load(config.RELATION_EMBEDDING)
        else:
            self.full_set = list(zip(words, textlen, positions, labels))
            self.train_set = list(zip(words_train, textlen_train, positions_train, labels_train)) # noqa
            self.valid_set = list(zip(words_valid, textlen_valid, positions_valid, labels_valid)) # noqa
            self.test_set = list(zip(words_test, textlen_test, positions_test, labels_test)) # noqa

        self.model_name = model_name
        self.runs = runs
        self.params_dict = params_dict
        self.hparams = AttrDict(params_dict)
        self.logger = logger

        self.model = self._get_model()
        self.saver = tf.train.Saver(tf.global_variables())
        checkpoint_dir = os.path.abspath(config.CHECKPOINT_DIR)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, self.__str__())

    def __str__(self):
        return self.model_name

    def _get_model(self):
        np.random.seed(config.RANDOM_SEED)
        kwargs = {
            "sequence_length": config.MAX_DOCUMENT_LENGTH,
            "num_classes": config.NUM_RELATION,
            "vocab_size": self.embedding.vocab_size,
            "embedding_size": self.embedding.embedding_dim,
            "position_size": self.embedding.position_size,
            "pretrained_embedding": self.embedding.embedding,
            "wpe": np.random.random_sample((self.embedding.position_size, self.hparams.wpe_size)),
            "hparams": self.hparams,
        }
        if "base" in self.model_name:
            return BiLSTM(**kwargs)
        elif "complex_hrere" in self.model_name:
            kwargs["entity_embedding1"] = self.entity_embedding1
            kwargs["entity_embedding2"] = self.entity_embedding2
            kwargs["relation_embedding1"] = self.relation_embedding1
            kwargs["relation_embedding2"] = self.relation_embedding2
            return ComplexHRERE(**kwargs)
        elif "real_hrere" in self.model_name:
            kwargs["entity_embedding"] = self.entity_embedding
            kwargs["relation_embedding"] = self.relation_embedding
            return RealHRERE(**kwargs)
        else:
            raise AttributeError("Invalid model name!")

    def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info("%s%s:" % (prefix, k))
                self.print_param_dict(v, prefix + incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix, k, v))

    def create_session(self):
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=8,
            allow_soft_placement=True,
            log_device_placement=False)
        return tf.Session(config=session_conf)

    def cv(self):
        self.logger.info("=" * 50)
        self.logger.info("Params")
        self._print_param_dict(self.params_dict)
        self.logger.info("Results")
        self.logger.info("\t\tRun\t\tStep\t\tLoss\t\tAcc\t\t\tAP")

        cv_loss = []
        cv_acc = []
        cv_ap = []
        for i in range(self.runs):
            sess = self.create_session()
            sess.run(tf.global_variables_initializer())
            step, loss, acc, ap = self.model.fit(sess, self.train_set, self.valid_set)
            self.logger.info("\t\t%d\t\t%d\t\t%.3f\t\t%.3f\t\t%.3f" %
                    (i + 1, step, loss, acc, ap))
            cv_loss.append(loss)
            cv_acc.append(acc)
            cv_ap.append(ap)
            sess.close()

        self.loss = np.mean(cv_loss)
        self.acc = np.mean(cv_acc)
        self.ap = np.mean(cv_ap)

        self.logger.info("CV Loss: %.3f" % self.loss)
        self.logger.info("CV Accuracy: %.3f" % self.acc)
        self.logger.info("CV Average Precision: %.3f" % self.ap)
        self.logger.info("-" * 50)

    def get_scores(self, labels, probs, if_save=False, prefix=""):
        average_precision = average_precision_score(labels, probs)

        order = np.argsort(-probs)

        top10 = order[:642]
        cnt10 = 0.0
        for i in top10:
            if labels[i] == 1:
                cnt10 += 1.0
        p10 = cnt10 / 642

        top30 = order[:642 * 3]
        cnt30 = 0.0
        for i in top30:
            if labels[i] == 1:
                cnt30 += 1.0
        p30 = cnt30 / 642 / 3

        top50 = order[:642 * 5]
        cnt50 = 0.0
        for i in top50:
            if labels[i] == 1:
                cnt50 += 1.0
        p50 = cnt50 / 642 / 5

        if if_save:
            if not os.path.exists(config.PLOT_OUT_DIR):
                os.makedirs(config.PLOT_OUT_DIR)
            np.save(os.path.join(config.PLOT_OUT_DIR, prefix + "_labels.npy"), labels)
            np.save(os.path.join(config.PLOT_OUT_DIR, prefix + "_probs.npy"), probs)
        return average_precision, p10, p30, p50

    def refit(self, prefix="", if_save=False):
        # self.logger.info("Params")
        # self._print_param_dict(self.params_dict)
        # self.logger.info("Evaluation for each epoch")
        # self.logger.info("\t\tEpoch\t\tAP\t\tAcc\t\tP@10%\t\tP@30%\t\tP@50%")

        sess = self.create_session()
        sess.run(tf.global_variables_initializer())
        epochs = 0
        # best_ap = 0.0
        # best_labels = None
        # best_probs = None
        probs_list = []
        for labels, probs, acc in self.model.evaluate(sess, self.full_set, self.test_set):
            epochs += 1
            # ap, p10, p30, p50 = self.get_scores(labels, probs)
            # self.logger.info("\t\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f" %
            #         (epochs, ap, acc, p10, p30, p50))
            # if best_ap < ap:
            #     best_ap = ap
            #     best_labels = labels
            #     best_probs = probs
            probs_list.append(probs_list)
        if len(probs_list) > 5:
            probs_list = probs_list[-5:]
        probs = np.mean(np.vstack(probs_list), axis=0)
        ap, p10, p30, p50 = self.get_scores(labels, probs)
        if if_save:
            self.model.save_preds(sess, self.test_set)
            self.get_scores(labels, probs, True, prefix)
        sess.close()
        return ap, p10, p30, p50

    def evaluate(self, prefix=""):
        self.logger.info("Params")
        self._print_param_dict(self.params_dict)
        self.logger.info("Final Evaluation")
        self.logger.info("-" * 50)

        aps = []
        p10s = []
        p30s = []
        p50s = []
        for i in range(self.runs):
            # sess = self.create_session()
            # sess.run(tf.global_variables_initializer())
            # self.model.fit(sess, self.full_set)
            # labels, probs, acc = self.model.predict(sess, self.test_set)
            # if i == 0:
            #     self.model.save_preds(sess, self.test_set)
            # ap, p10, p30, p50 = self.get_scores(labels, probs)
            # sess.close()
            ap, p10, p30, p50 = self.refit(prefix, i == 0)

            aps.append(ap)
            p10s.append(p10)
            p30s.append(p30)
            p50s.append(p50)

            self.logger.info("PR curve area: %.3f" % ap)
            self.logger.info("P@10%%: %.3f" % p10)
            self.logger.info("P@30%%: %.3f" % p30)
            self.logger.info("P@50%%: %.3f" % p50)
            self.logger.info("-" * 50)

        # probs = np.mean(np.vstack(probs_list), axis=0)
        # ap, p10, p30, p50 = self.get_scores(labels, probs, True, prefix)
        self.logger.info("Average Results")
        self.logger.info("PR curve area: %.3f(+-%.3f)" % (np.mean(aps), np.std(aps)))
        self.logger.info("P@10%%: %.3f(+-%.3f)" % (np.mean(p10s), np.std(p10s)))
        self.logger.info("P@30%%: %.3f(+-%.3f)" % (np.mean(p30s), np.std(p30s)))
        self.logger.info("P@50%%: %.3f(+-%.3f)" % (np.mean(p50s), np.std(p50s)))
        self.logger.info("=" * 50)


class TaskOptimizer:
    def __init__(self, model_name, max_evals, runs, logger):
        self.model_name = model_name
        self.max_evals = max_evals
        self.runs = runs
        self.logger = logger
        self.model_param_space = ModelParamSpace(self.model_name)

    def _obj(self, param_dict):
        param_dict = self.model_param_space._convert_into_param(param_dict)
        self.task = Task(self.model_name, self.runs, param_dict, self.logger)
        self.task.cv()
        tf.reset_default_graph()
        ret = {
            "loss": -self.task.ap,
            "attachments": {
                "loss": self.task.loss,
                "acc": self.task.acc,
            },
            "status": STATUS_OK
        }
        return ret

    def run(self):
        trials = Trials()
        best = fmin(self._obj, self.model_param_space._build_space(),
                tpe.suggest, self.max_evals, trials)
        best_params = space_eval(self.model_param_space._build_space(), best)
        best_params = self.model_param_space._convert_into_param(best_params)
        trial_loss = np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_loss)
        best_ap = trial_loss[best_ind]
        best_loss = trials.trial_attachments(trials.trials[best_ind])["loss"]
        best_acc = trials.trial_attachments(trials.trials[best_ind])["acc"]
        self.logger.info("-" * 50)
        self.logger.info("Best Average Precision: %.3f" % best_ap)
        self.logger.info("with Loss %.3f, Accuracy %.3f" % (best_loss, best_acc))
        self.logger.info("Best Param:")
        self.task._print_param_dict(best_params)
        self.logger.info("-" * 50)


def parse_args(parser):
    parser.add_option("-m", "--model", type="string", dest="model_name", default="base")
    parser.add_option("-e", "--eval", type="int", dest="max_evals", default=100)
    parser.add_option("-r", "--runs", type="int", dest="runs", default=3)
    options, args = parser.parse_args()
    return options, args


def main(options):
    time_str = datetime.datetime.now().isoformat()
    logname = "[Model@%s]_%s.log" % (options.model_name, time_str)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    optimizer = TaskOptimizer(options.model_name, options.max_evals, options.runs, logger)
    optimizer.run()


if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
