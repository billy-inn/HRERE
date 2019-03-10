
# ---------------------- PATH ----------------------
ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
KG_PATH = "%s/kg" % ROOT_PATH

CHECKPOINT_DIR = "%s/checkpoint" % ROOT_PATH
LOG_DIR = "%s/log" % ROOT_PATH
PLOT_DIR = "%s/plot" % ROOT_PATH
PLOT_DATA_DIR = "%s/data" % PLOT_DIR
PLOT_OUT_DIR = "%s/output" % PLOT_DIR
PLOT_FIG_DIR = "%s/figure" % PLOT_DIR

# ---------------------- DATA ----------------------
RAW_TRAIN_DATA = "%s/train.txt" % DATA_PATH
CLEAN_TRAIN_DATA = "%s/train.tsv" % DATA_PATH
GROUPED_TRAIN_DATA = "%s/grouped_train.pkl" % DATA_PATH

RAW_TEST_DATA = "%s/test.txt" % DATA_PATH
CLEAN_TEST_DATA = "%s/test.tsv" % DATA_PATH
GROUPED_TEST_DATA = "%s/grouped_test.pkl" % DATA_PATH

EMBEDDING_DATA = "%s/glove.840B.300d.txt" % DATA_PATH

E2ID = "%s/entity2id.txt" % DATA_PATH
R2ID = "%s/relation2id.txt" % DATA_PATH

# TransE
ENTITY_EMBEDDING = "%s/entity.npy" % DATA_PATH
RELATION_EMBEDDING = "%s/relation.npy" % DATA_PATH
# Complex
ENTITY_EMBEDDING1 = "%s/entity1.npy" % DATA_PATH
ENTITY_EMBEDDING2 = "%s/entity2.npy" % DATA_PATH
RELATION_EMBEDDING1 = "%s/relation1.npy" % DATA_PATH
RELATION_EMBEDDING2 = "%s/relation2.npy" % DATA_PATH

KB = "%s/triples.csv" % DATA_PATH

# ---------------------- PARAM ---------------------
MAX_DOCUMENT_LENGTH = 100

NUM_RELATION = 55

BAG_SIZE = 5

RANDOM_SEED = 2019
