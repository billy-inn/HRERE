import config
import numpy as np
from utils.data_utils import load_dict_from_txt
from optparse import OptionParser

def real():
    entity = np.load(config.KG_PATH + "/entity.npy")
    relation = np.load(config.KG_PATH + "/relation.npy")
    e2id = load_dict_from_txt(config.KG_PATH + "/e2id.txt")
    r2id = load_dict_from_txt(config.KG_PATH + "/r2id.txt")

    entity2id = load_dict_from_txt(config.E2ID)
    relation2id = load_dict_from_txt(config.R2ID)
    e_embeddings = np.random.uniform(0.0, 1.0, (len(entity2id), entity.shape[1]))
    r_embeddings = np.random.uniform(0.0, 1.0, (len(relation2id), relation.shape[1]))
    for e in entity2id:
        if e not in e2id:
            continue
        idx1 = entity2id[e]
        idx2 = e2id[e]
        e_embeddings[idx1, :] = entity[idx2, :]
    for r in relation2id:
        if r not in r2id:
            continue
        idx1 = relation2id[r]
        idx2 = r2id[r]
        r_embeddings[idx1, :] = relation[idx2, :]
    np.save(config.ENTITY_EMBEDDING, e_embeddings)
    np.save(config.RELATION_EMBEDDING, r_embeddings)

def complex():
    entity1 = np.load(config.KG_PATH + "/entity1.npy")
    entity2 = np.load(config.KG_PATH + "/entity2.npy")
    relation1 = np.load(config.KG_PATH + "/relation1.npy")
    relation2 = np.load(config.KG_PATH + "/relation2.npy")
    e2id = load_dict_from_txt(config.KG_PATH + "/e2id.txt")
    r2id = load_dict_from_txt(config.KG_PATH + "/r2id.txt")

    entity2id = load_dict_from_txt(config.E2ID)
    relation2id = load_dict_from_txt(config.R2ID)
    e1_embeddings = np.random.uniform(0.0, 1.0, (len(entity2id), entity1.shape[1]))
    e2_embeddings = np.random.uniform(0.0, 1.0, (len(entity2id), entity2.shape[1]))
    r1_embeddings = np.random.uniform(0.0, 1.0, (len(relation2id), relation1.shape[1]))
    r2_embeddings = np.random.uniform(0.0, 1.0, (len(relation2id), relation2.shape[1]))
    for e in entity2id:
        if e not in e2id:
            continue
        idx1 = entity2id[e]
        idx2 = e2id[e]
        e1_embeddings[idx1, :] = entity1[idx2, :]
        e2_embeddings[idx1, :] = entity2[idx2, :]
    for r in relation2id:
        if r not in r2id:
            continue
        idx1 = relation2id[r]
        idx2 = r2id[r]
        r1_embeddings[idx1, :] = relation1[idx2, :]
        r2_embeddings[idx1, :] = relation2[idx2, :]
    np.save(config.ENTITY_EMBEDDING1, e1_embeddings)
    np.save(config.ENTITY_EMBEDDING2, e2_embeddings)
    np.save(config.RELATION_EMBEDDING1, r1_embeddings)
    np.save(config.RELATION_EMBEDDING2, r2_embeddings)

def parse_args(parser):
    parser.add_option("-e", "--emb_type", type="string", dest="emb_type", default="complex")
    options, args = parser.parse_args()
    return options, args

def main(options):
    if options.emb_type == "complex":
        complex()
    else:
        real()

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
