from keras.utils import to_categorical
import numpy as np
import pandas as pd
import subprocess
from keras.utils import Sequence
import re

def rev_comp(seq):
    # =============================================================================
    #     This function provides the revers-complement of the input sequence
    #     Input: seq - sequence string
    #     Output: rc - reverse complement sequence string
    # =============================================================================
    trantab = str.maketrans('ACGTN', 'TGCAN')
    rc = seq.translate(trantab)[::-1]
    return rc


def read_chr(path):
    # =============================================================================
    #   This function read a chromosome from the provided path.
    #     Input: ptath - input file path
    #     Output: chr - chromosome string
    # =============================================================================
    chr_n = pd.read_csv(path)
    i = path[11:-3]
    chr = chr_n['>chr' + i].str.cat(sep='')

    return chr


def get_ds(path):

    sp = "wc -l " + path + " | awk '{print $1}'"
    pr = subprocess.Popen(sp, stdout=subprocess.PIPE, shell=True)
    out = pr.stdout.read()
    out = out.decode("utf-8")

    return int(out[:-2])

def padding(mat, max_seq):
    # =============================================================================
    #     This function creates zero padding by conctinating it with a matrix of zeros
    #     Input: mat - the original one hot matrix, max_seq - the final width of the matrix
    #     Output: concate - the zerro padded matrix
    # =============================================================================
    l1 = int(np.floor((max_seq - mat.shape[0]) / 2))
    l2 = int(max_seq - mat.shape[0] - l1)
    ze1 = np.zeros((l1, mat.shape[1]), dtype='int')
    ze2 = np.zeros((l2, mat.shape[1]), dtype='int')
    concate = np.concatenate((ze1, mat, ze2), axis=0)
    return concate


def oneHot(string, max_seq=None):
    # =============================================================================
    #   This function creats one hot encoding to a given DNA string
    #     Input: string - the DNA string, max_seq - final length of the sequence
    #     Output: mat - the encoded matrix
    # =============================================================================
    trantab = str.maketrans('ACGTN', '01234')
    string = string + 'ACGTN'
    data = list(string.translate(trantab))
    mat = to_categorical(data)[0:-5]
    mat = np.delete(mat, -1, -1)

    # if max_seq and len(string) < max_seq:
    if len(string) - 5 < max_seq:
        mat = padding(mat, max_seq)

    mat = mat.astype(np.uint8)

    return mat



def read_seq(df, gn, f):
    # =============================================================================
    #   This extracts DNA sequence from genome assembly according to provided bedGraph coordinates
    #     Input: df - DataFrame object with sequence information, gn - list of chromosome strings,
    #            f - size of padding flanks
    #     Output: seq - extracted sequence padded with flanks
    # =============================================================================
    chro = df['chr'][:][3:]
    if chro == 'X':
        chro = 22
    elif chro == 'Y':
        chro = 23
    elif chro == 'M':
        chro = 24
    else:
        chro = int(chro)-1

    start = df['start'] - f

    if start < 0:
        start = df['start']

    end = df['end'] + f
    if end > len(gn[chro]):
        end = df['end']

    seq = gn[chro][start:end].upper()

    if df['mp'] == 1:
        seq = rev_comp(seq)

    return seq

def toFolds(fold, start, x, y=None):
    # =============================================================================
    #   This function splits a given dataset into test and train sets
    #     Input: x - a list of numpy arrays of the model inputs
    #            y - numpy array the complete set of labels
    #            fold - the size of the test set
    #            start - the index in the entire set where the test set starts
    #     Output: x_test, y_test, x_train, y_train
    # =============================================================================

    x_train = []
    x_test = []
    if start + fold > len(x):
        end = len(x)
    else:
        end = start + fold

    for X in x:
        x_test.append(X[start:end].copy())

        x_train.append(np.delete(X, range(start, end), axis=0))

    if len(x_test) == 0:
        x_test = x_test[0]
        x_train = x_train[0]

    if y:  # when labels are provided
        y_test = y[start:end].copy()
        y_train = np.delete(x, range(start, end))

        return x_test, y_test, x_train, y_train

    else:
        return x_test, x_train


def loop_info(loop):

    split_l = re.split(r'\[|\]|,| ', loop)
    split_l = list(filter(lambda x: x != '', split_l))
    ll = np.array(split_l, dtype=np.uint8)

    return ll



class MissGen(Sequence):

    def __init__(self, ind_co, bs, chro, path, stat, flank, shuffle=True, train=True):

        self.ind_co = ind_co
        self.shuffle = shuffle
        self.bs = bs
        self.path = path
        self.chr = chro
        self.train = train
        self.stat = stat
        self.locs = pd.read_csv(self.path, header=None, names=["chr", "start", "end", "mm", "mp"]) #sep='\t'
        self.f = flank

    def __len__(self):
        return int(np.ceil(len(self.ind_co) / self.bs))

    def __getitem__(self, ind):

        inds = self.ind_co[ind*self.bs:np.min([(ind+1)*self.bs, len(self.ind_co)])]

        X, y = self.__data_gen(inds)

        if self.train:
            return X, y
        else:
            return X

    def on_epoch_end(self):

        if self.shuffle:
            self.locs = self.locs.sample(frac=1).reset_index(drop=True)
            print('data shuffled')

    def get_seq(self, data_temp):

        data_temp['seq'] = data_temp.apply(lambda x: read_seq(x, self.chr, self.f), axis=1)

        return data_temp.reset_index(drop=True)

    def __data_gen(self, ind_temp):

        data_temp = self.get_seq(self.locs[ind_temp[0]:ind_temp[-1]+1])
        X = list(data_temp['seq'].apply(lambda x: oneHot(x, 15 + (self.f*2))))
        X = np.asarray(X)

        if len(X.shape) < 3:
            print(X.shape)
            data_temp.to_csv('wrong.csv')
        y = data_temp['mm']

        return X, y



