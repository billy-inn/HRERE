import pandas as pd
import numpy as np
from utils import pkl_utils
from optparse import OptionParser
import config
from utils.data_utils import load_dict_from_txt

def word_preprocess(w):
    if w == '-lrb-':
        return ["``"]
    if w == '-rrb-':
        return ["''"]
    if any(c.isalpha() for c in w):
        filters = '!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
        w = w.translate(str.maketrans(filters, ' ' * len(filters)))
        return w.split()
    if w in "#$%&*+=@^|":
        return []
    return [w]

def search(s, t, offset=0):
    for i in range(offset, len(s) - len(t) + 1):
        flag = True
        for j in range(len(t)):
            try:
                if s[i + j] != t[j]:
                    flag = False
                    break
            except Exception:
                print(s, t)
        if flag:
            return i, i + len(t) - 1
    return -1, -1

def find_pos(info):
    s, e1, e2 = info.split("\t")
    tokens = s.split()
    w1 = word_preprocess(e1)
    w2 = word_preprocess(e2)
    words = []
    x1 = y1 = x2 = y2 = -1
    for i in range(len(tokens)):
        if tokens[i] == e1:
            if x1 == -1:
                x1 = len(words)
                y1 = x1 + len(w1) - 1
            words.extend(w1)
            continue
        if tokens[i] == e2:
            if x2 == -1:
                x2 = len(words)
                y2 = x2 + len(w2) - 1
            words.extend(w2)
            continue
        words.extend(word_preprocess(tokens[i]))
    if x1 == -1:
        x1, y1 = search(words, w1)
        if (x2 >= x1) and (y2 <= y1):
            x2, y2 = search(words, w2, y2)
    if x2 == -1:
        x2, y2 = search(words, w2)
        if (x1 >= x2) and (y1 <= y2):
            x1, y1 = search(words, w1, y1)
    return " ".join(words), x1, y1, x2, y2

def preprocess(raw_data, clean_data, if_test=False):
    e2id = load_dict_from_txt(config.E2ID)
    r2id = load_dict_from_txt(config.R2ID)

    if if_test:
        df = pd.read_csv(raw_data, sep="\t", names=["a1", "a2", "e1", "e2", "r", "s", "end"],
                na_values=[], keep_default_na=False)
    else:
        df = pd.read_csv(raw_data, sep="\t", names=["a1", "a2", "e1", "e2", "r", "s"],
                na_values=[], keep_default_na=False)
        df.s = df.s.map(lambda x: " ".join(x.split()[:-1]))

    df["len1"] = df.e1.map(lambda x: len(x.split('_')))
    df["len2"] = df.e2.map(lambda x: len(x.split('_')))
    df["len"] = df.s.map(lambda x: len(x.split())) + df.len1 + df.len2 - 2

    df["info"] = df.s + "\t" + df.e1 + "\t" + df.e2
    df["info"] = df["info"].map(lambda x: find_pos(x))
    df["s"] = df["info"].map(lambda x: x[0])
    df["x1"] = df["info"].map(lambda x: x[1])
    df["y1"] = df["info"].map(lambda x: x[2])
    df["x2"] = df["info"].map(lambda x: x[3])
    df["y2"] = df["info"].map(lambda x: x[4])
    df["e1"] = df.a1.map(e2id)
    df["e2"] = df.a2.map(e2id)

    def transform(x):
        if x == "/business/company/industry":
            return "/business/business_operation/industry"
        if x == "/business/company/locations":
            return "/organization/organization/locations"
        if x == "/business/company/founders":
            return "/organization/organization/founders"
        if x == "/business/company/major_shareholders":
            return "/organization/organization/founders"
        if x == "/business/company/advisors":
            return "/organization/organization/advisors"
        if x == "/business/company_shareholder/major_shareholder_of":
            return "/organization/organization_founder/organizations_founded"
        if x == "business/company/place_founded":
            return "/organization/organization/place_founded"
        if x == "/people/person/place_lived":
            return "/people/person/place_of_birth"
        if x == "/business/person/company":
            return "/organization/organization_founder/organizations_founded"
        return x

    df.r = df.r.map(transform)
    df.r = df.r.map(lambda x: r2id.get(x, 0)).astype(int)
    print(df.r.value_counts())

    df[["r", "e1", "x1", "y1", "e2", "x2", "y2", "s"]].to_csv(clean_data,
            sep="\t", index=False, header=False)

def group(input_data, output_data, if_sample=False):
    df = pd.read_csv(input_data, sep="\t",
            names=["r", "e1", "x1", "y1", "e2", "x2", "y2", "s"])
    grouped = df.groupby(["r", "e1", "e2"])
    words = []
    positions = []
    heads = []
    tails = []
    labels = []
    cnt = 0
    for name, group in grouped:
        if if_sample and cnt > 10000:
            break
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        group = group.reset_index(drop=True)
        label = name[0]
        head = name[1]
        tail = name[2]
        size = group.shape[0]
        tmp_words = []
        tmp_positions = []
        for i in range(size):
            tmp_words.append(group.s[i])
            tmp_positions.append([group.x1[i], group.y1[i], group.x2[i], group.y2[i]])
        if size < config.BAG_SIZE:
            tmp = size
            ans_words = tmp_words[:]
            ans_positions = tmp_positions[:]
            while tmp + size < config.BAG_SIZE:
                tmp += size
                ans_words += tmp_words
                ans_positions += tmp_positions
            ans_words += tmp_words[:config.BAG_SIZE - tmp]
            ans_positions += tmp_positions[:config.BAG_SIZE - tmp]
            words.append(ans_words)
            positions.append(ans_positions)
            heads.append(head)
            tails.append(tail)
            labels.append(label)
        else:
            tmp = 0
            while tmp + config.BAG_SIZE < size:
                words.append(tmp_words[tmp:tmp + config.BAG_SIZE])
                positions.append(tmp_positions[tmp:tmp + config.BAG_SIZE])
                heads.append(head)
                tails.append(tail)
                labels.append(label)
                tmp += config.BAG_SIZE
            words.append(tmp_words[-config.BAG_SIZE:])
            positions.append(tmp_positions[-config.BAG_SIZE:])
            heads.append(head)
            tails.append(tail)
            labels.append(label)
    heads = np.array(heads)
    tails = np.array(tails)
    labels = np.array(labels)
    pkl_utils._save(output_data, (words, positions, heads, tails, labels))

def create_test_set(input_data, output_data, if_sample):
    df = pd.read_csv(input_data, sep="\t",
            names=["r", "e1", "x1", "y1", "e2", "x2", "y2", "s"])
    words = []
    positions = []
    heads = []
    tails = []
    labels = []
    size = df.shape[0]
    cnt = 0
    for i in range(size):
        if if_sample and cnt > 10000:
            break
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        tmp_words = []
        tmp_positions = []
        for _ in range(config.BAG_SIZE):
            tmp_words.append(df.s[i])
            tmp_positions.append([df.x1[i], df.y1[i], df.x2[i], df.y2[i]])
        words.append(tmp_words)
        positions.append(tmp_positions)
        heads.append(df.e1[i])
        tails.append(df.e2[i])
        labels.append(df.r[i])
    heads = np.array(heads)
    tails = np.array(tails)
    labels = np.array(labels)
    pkl_utils._save(output_data, (words, positions, heads, tails, labels))

def parse_args(parser):
    parser.add_option("-p", default=False, action="store_true", dest="preprocess")
    parser.add_option("-g", default=False, action="store_true", dest="group")
    parser.add_option("-s", default=False, action="store_true", dest="sample")

    options, args = parser.parse_args()
    return options, args

def main(options):
    if options.preprocess:
        preprocess(config.RAW_TRAIN_DATA, config.CLEAN_TRAIN_DATA)
        preprocess(config.RAW_TEST_DATA, config.CLEAN_TEST_DATA, True)
    if options.group:
        group(config.CLEAN_TRAIN_DATA, config.GROUPED_TRAIN_DATA, options.sample)
        create_test_set(config.CLEAN_TEST_DATA, config.GROUPED_TEST_DATA, options.sample)

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
