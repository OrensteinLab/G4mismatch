from tensorflow import keras
import numpy as np
from mm_utils import oneHot, padding
from joblib import Parallel, delayed


class MissGen(keras.utils.Sequence):

    def __init__(self, bs, genome, locs, stat, flank, shuffle=True, train=True, split=False, val_N=None):

        self.shuffle = shuffle
        self.bs = bs
        self.chr = Parallel(n_jobs=8)(delayed(oneHot)(str(x.seq)) for x in genome)
        self.train = train
        self.stat = stat
        self.locs = locs
        self.f = flank
        self.ind_co = np.arange(len(self.locs))
        self.chr_list = [x.name for x in genome]
        self.split = split
        self.n_data = len(locs)
        self.valN = val_N
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n_data / self.bs))

    def __getitem__(self, ind):

        inds = self.ind_co[ind * self.bs:np.min([(ind + 1) * self.bs, len(self.ind_co)])]

        X, y = self.__data_gen(inds)

        if self.split:
            X = [X[:, :self.f, :], X[:, self.f:-self.f, :], X[:, -self.f:, :]]

        if self.train:
            return X, y
        else:
            return X

    def on_epoch_end(self):

        if self.shuffle:
            self.locs = self.locs.sample(frac=1).reset_index(drop=True)
            print('\ndata shuffled')


    def read_seq(self, df):
        
        flag1 = False
        flag2 = False 

        chr_ind = self.chr_list.index(df['chr'])

        start = df['start'] - self.f
        if start < 0:
            flag1 = True
            pad_side = 'left'
            start = 0
        
        end = df['end'] + self.f
        if end > len(self.chr[chr_ind]):
            flag2 = True
            pad_side = 'right'
            end = len(self.chr[chr_ind])

        seq = self.chr[chr_ind][int(start):int(end), :].copy()

        if flag1 or flag2:
            seq = padding(seq, self.f * 2 + 15, pad_side=pad_side)

        if df['mp'] == '+':
            seq = seq[::-1, ::-1]

        if seq.shape != (self.f*2 + 15, 4):
            print(seq.shape)
            print(df)
            
        return seq

    def get_seq(self, data_temp):

        data_temp['oh'] = data_temp.apply(self.read_seq, axis=1)

        return data_temp.reset_index(drop=True)

    def __data_gen(self, ind_temp):

        data_temp = self.get_seq(self.locs[ind_temp[0]:ind_temp[-1] + 1])
        try:
            X = np.stack(data_temp['oh'].values)
        except Exception as e:
            print(e)
            
        y = data_temp['mm'].values

        if self.valN != None:
            ind = np.where(~X.any(axis=-1))
            X[ind[0], ind[1], :] = self.valN

        return X, y



class GenScan(keras.utils.Sequence):

    def __init__(self, win_size, bs, seq, flank, rc=False):

        self.bs = bs
        self.f = flank
        self.win_size = win_size
        self.f = flank
        self.seq = seq
        self.rc = rc

    def __len__(self):
        return int(np.ceil((len(self.seq) - self.f * 2 - self.win_size) / self.bs))

    def __getitem__(self, ind):

        start = ind * self.bs
        end = start + self.f * 2 + self.win_size + self.bs
        if end > len(self.seq):
            num_win = self.bs - (end - len(self.seq)) + 1
            if num_win < 1:
                num_win = 1
            end = len(self.seq)
        else:
            num_win = self.bs
        print('workig on batch {:d}, starting at ind={:d} and ending at ind={:d}'.format(ind, start, end))

        X = self.__data_gen(start, end, num_win=num_win)

        return X

    def __data_gen(self, start, end, num_win):

        arr = self.seq[start:end, :].copy()

        if self.rc:
            win_arr = np.stack([arr[i:i + self.win_size + self.f * 2, :][::-1, ::-1] for i in range(num_win)])
        else:
            win_arr = np.stack([arr[i:i + self.win_size + self.f * 2, :] for i in range(num_win)])

        return win_arr

