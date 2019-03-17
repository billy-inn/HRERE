# HRERE

Connecting Language and Knowledge with Heterogeneous Representations for Neural Relation Extraction

Paper Published in NAACL 2019: [NFETC](https://arxiv.org/abs/1803.03378)

### Prerequisites

- tensorflow >= r1.2
- hyperopt
- gensim
- sklearn

### Dataset

### Preprocessing

Construct the knowledge graph:

```
python create_kg.py
```

Preprocessing the data:

```
python preprocess.py -p -g
```

### Complex Embeddings

Copy the fils in `./kg` to the data folder in [tensorflow-efe](https://github.com/billy-inn/tensorflow-efe) and run the following commands to obtain the complex embeddings:

```
python train.py --model best_Complex_tanh_fb3m --data fb3m --save
python get_embeddings.py --embed complex --model best_Complex_tanh_fb3m --output <current_data_path>
```

### Hyperparameters Tuning

```
python task.py --model <model_name> --eval <max_number_of_search> --runs <number_of_runs_per_setting>
```

`model_name` can be found in `model_param_space.py`. You can define the search space by yourself.

### Evaluation

```
python eval.py --model <model_name> --prefix <file_prefix> --runs <number_of_runs> [--eval]
```

`model_name` can be found in `model_param_space.py`.
Option `--eval` is used to save the data for plotting; otherwise, it'll run multiple times and calculate the means and stds of P@N. 

### Cite

If you found this codebase or our work useful, please cite:

```
@InProceddings{xu2019connecting,
  author = {Xu, Peng and Barbosa, Denilson},
  title = {Connecting Language and Knowledge with Heterogeneous Representations for Neural Relation Extraction}
  booktitle = {The 17th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2019)},
  month = {June},
  year = {2019},
  publisher = {ACL}
}
```
