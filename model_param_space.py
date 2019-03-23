import numpy as np
from hyperopt import hp

param_space_base = {
    "wpe_size": hp.quniform("wpe_size", 5, 100, 5),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-3), 1e-4),
    "state_size": hp.quniform("state_size", 100, 500, 10),
    "hidden_layers": 1,
    "hidden_size": hp.quniform("hidden_size", 100, 500, 10),
    "dense_keep_prob": hp.quniform("dense_keep_prob", 0.5, 1, 0.1),
    "rnn_keep_prob": hp.quniform("rnn_keep_prob", 0.5, 1, 0.1),
    "l2_reg_lambda": hp.quniform("l2_reg_lambda", 0, 1e-3, 1e-4),
    "batch_size": 128,
    "num_epochs": 20,
}

param_space_best_base = {
    "wpe_size": 25,
    "lr": 0.0005,
    "state_size": 320,
    "hidden_layers": 1,
    "hidden_size": 190,
    "dense_keep_prob": 0.9,
    "rnn_keep_prob": 0.7,
    "l2_reg_lambda": 0.0003,
    "batch_size": 128,
    "num_epochs": 30,
}

param_space_complex = {
    "wpe_size": 25,
    "lr": 0.0005,
    "lr2": 1e-6,
    "state_size": 320,
    "hidden_layers": 1,
    "hidden_size": 190,
    "dense_keep_prob": 0.9,
    "rnn_keep_prob": 0.7,
    "l2_reg_lambda": 0.0003,
    "batch_size": 128,
    "num_epochs": 30,
    "alpha": hp.quniform("alpha", 0.5, 0.9, 0.05),
    "lambda1": hp.quniform("lambda1", 0.1, 2.0, 0.1),
    "lambda2": hp.quniform("lambda2", 0.1, 2.0, 0.1),
}

param_space_best_complex = {
    "wpe_size": 25,
    "lr": 0.0005,
    "lr2": 1e-6,
    "state_size": 320,
    "hidden_layers": 1,
    "hidden_size": 190,
    "dense_keep_prob": 0.9,
    "rnn_keep_prob": 0.7,
    "l2_reg_lambda": 0.0003,
    "batch_size": 128,
    "num_epochs": 30,
    "alpha": 0.9,
    "lambda1": 1.0,
    "lambda2": 1.0,
}

param_space_real = {
    "wpe_size": 25,
    "lr": 0.0005,
    "lr2": 1e-6,
    "state_size": 320,
    "hidden_layers": 1,
    "hidden_size": 190,
    "dense_keep_prob": 0.9,
    "rnn_keep_prob": 0.7,
    "l2_reg_lambda": 0.0003,
    "batch_size": 128,
    "num_epochs": 30,
    "alpha": hp.quniform("alpha", 0.5, 0.9, 0.05),
    "lambda1": hp.quniform("lambda1", 0.1, 2.0, 0.1),
    "lambda2": hp.quniform("lambda2", 0.1, 2.0, 0.1),
}

param_space_best_real = {
    "wpe_size": 25,
    "lr": 0.0005,
    "lr2": 1e-6,
    "state_size": 320,
    "hidden_layers": 1,
    "hidden_size": 190,
    "dense_keep_prob": 0.9,
    "rnn_keep_prob": 0.7,
    "l2_reg_lambda": 0.0003,
    "batch_size": 128,
    "num_epochs": 30,
    "alpha": 0.9,
    "lambda1": 1.0,
    "lambda2": 1.0,
}

param_space_dict = {
    "base": param_space_base,
    "best_base": param_space_best_base,
    "complex_hrere": param_space_complex,
    "best_complex_hrere": param_space_best_complex,
    "real_hrere": param_space_real,
    "best_real_hrere": param_space_best_real,
}

int_params = [
    "wpe_size", "state_size", "batch_size", "num_epochs",
    "hidden_size", "hidden_layers",
]


class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Invalid model name!"
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_into_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_into_param(v[i])
                elif isinstance(v, dict):
                    self._convert_into_param(v)
        return param_dict
