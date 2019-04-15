from multiprocessing import Pool
import os, time, random
from tqdm import *
import os.path as osp
from dataset.tfrecord_generator import DataGenerator
from lang_dict.lang_dict import LanguageDict

if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())

    root = "/home/chengli/data/fine_data"
    files = ["annotation_train.txt"]

    lines = []
    for file in files:
        with open(osp.join(root, file), "r") as f:
            lines += f.readlines()
    lines = [osp.join(root, line.strip()) for line in tqdm(lines)]

    lang_dict = LanguageDict()

    BATCH = 128 * 256

    N_BATCH = len(lines) // BATCH
    p = Pool(12)
    for i in tqdm(
            p.imap_unordered(DataGenerator.generator_by_tuple,
                             zip([i for i in range(N_BATCH)],
                                 [lang_dict for _ in range(N_BATCH)],
                                 [lines[i * BATCH: (i + 1) * BATCH] for i in range(BATCH)]))
            , total=N_BATCH):
        pass
    p.terminate()
    print('All subprocesses done.')
