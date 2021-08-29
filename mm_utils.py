import pandas as pd
import numpy as np
from Bio import SeqIO

def rev_comp(seq):
    trantab = str.maketrans('ACGTN', 'TGCAN')
    return seq.translate(trantab)[::-1]


def read_chr(path):
    chr_n = pd.read_csv(path, skiprows=1, header=None)
    chr = chr_n[0].str.cat(sep='')
    return chr.upper()

def read_genome(path, chr_list=None):

    genome = list(SeqIO.parse(path, 'fasta'))
    if chr_list:
        chroms = [x.upper() for x in genome if x.name in chr_list]
    else:
        chroms = [x.upper() for x in genome]

    return chroms

def padding(mat, max_seq, val = 0, pad_side='around'):
    if pad_side == 'around':
        l1 = int(np.floor((max_seq - mat.shape[0]) / 2))
        l2 = int(max_seq - mat.shape[0] - l1)
        ze1 = np.ones((l1, mat.shape[1]), dtype='int')*val
        ze2 = np.ones((l2, mat.shape[1]), dtype='int')*val
        concate = np.concatenate((ze1, mat, ze2), axis=0)
    else:
        l = int(max_seq - mat.shape[0])
        ze = np.zeros((l, mat.shape[1]), dtype='int')
        if pad_side == 'right':
            concate = np.concatenate((mat, ze), axis=0)
        elif pad_side == 'left':
            concate = np.concatenate((ze, mat), axis=0)

    return concate


def oneHot(string, max_seq=None, pad_side='around', val_N=0):
    mat = np.zeros((len(string), 5), dtype=np.int8)
    trantab = str.maketrans('AaCcGgTtNYRWSKMBDHV', '0011223344444444444')
    data = list(string.translate(trantab))
    data_arr = np.array(data, dtype=np.int8)
    mat[range(data_arr.size), data_arr] = 1
    if val_N:
        mat = mat.astype(np.float16)
        ind = np.where(mat[:, -1] == 1)[0]
        mat[ind, :] = val_N
    mat = mat[:, :-1]

    if max_seq and len(string) < max_seq:
        mat = padding(mat, max_seq, val_N, pad_side)

    return mat
