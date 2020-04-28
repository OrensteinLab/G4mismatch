from utilsGGG import *
from model import model as mdl
import pickle
import sys
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import argparse
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def user_input():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stabilizer', help='Stabilizing molecule.\n '
                                                   'We used K and PDS, but you may choose otherwise.',
                        choices=['K', 'PDS'], type=str, required=False, default='K')
    parser.add_argument('-f', '--flank', help='Sequence padding flank lengths.\n '
                                              'We used the 0, 50,100 and 150, but you may choose otherwise',
                        type=str, required=False, default='100')
    parser.add_argument('-of', '--output_file', help='Name of test results output file', type=str,
                        required=False, default='G4mismatchWG_out.csv')
    parser.add_argument('-trf', '--train_file', help='Path to training dataset of type bedGraph or FASTA',
                        type=str, required=False, default=r'data\GSE63874_Na_K_train.bedGraph')
    parser.add_argument('-tf', '--test_file', help='Output file name', type=str,
                        required=False, default=r'K_chr1_test.bedGraph')
    parser.add_argument('-u', '--use', help='Use G4mismatch for testing on your own data or retrain with a new dataset',
                        choices=['train', 'test'], type=str, required=False, default='test')
    parser.add_argument('-g', '--genome_path', help='Path to the genome assembly of interest', type=str, required=False)
    parser.add_argument('-ug', '--use_generator', help='Boolean indicted if training is to be done with a generator',
                        choices=['True', 'False'], type=str, required=False, default='False')
    parser.add_argument('-e', '--epochs', help='Number of training epochs', type=str, required=False, default='50')
    parser.add_argument('-bs', '--batch_size', help='Size of training batch', type=str, required=False, default='1000')
    parser.add_argument('-w', '--workers', help='Maximum number of processes to spin for generator',
                        type=str, required=False, default='40')
    parser.add_argument('-q', '--queue', help='Maximum queue size for generator',
                        type=str, required=False, default='100')
    parser.add_argument('-sc', '--scores', help='Target scores for fasta input in training mode',
                        type=str, required=False)
    parser.add_argument('-m', '--model_path', help='Path to required model for test mode',
                        type=str, required=False, default='50')
    parser.add_argument('-nj', '--number_of_jobs', help='Number of jobs for reading genome assembly',
                        type=str, required=False, default='1')


    args = parser.parse_args()
    arguments = vars(args)

    return arguments



def main():

    args = user_input()

    stab = args['stabilizer']
    flank = args['flank']


    if args['use'] == 'train':

        tr_ext = os.path.splitext(args['train_file'])[1]
        tr_p = os.path.split(args['output_file'])[0]
        if tr_p != '':
            tr_p = tr_p + '/'

        if not tr_ext == '.bedGraph' or not tr_ext == '.fa':
            print('allowed input file extensions are bedGraph or fa')
            sys.exit()

        if tr_ext == '.bedGraph':
            if not args['genome_path']:
                print('use -g to provide path to relevant genome assembly')
                sys.exit()
            else:
                l = list(np.arange(1, 23)) + ['X', 'Y', 'M']
                paths = list(map(lambda x: '../hg19/chr' + str(x) + '.fa', l))
                gn = Parallel(n_jobs=args['number_of_jobs'])(delayed(read_chr)(x) for x in paths)

        model = mdl((15 + (flank * 2), 4), filters=256, opt_func='Adam', lr=1e-3, fc=32)
        mc = ModelCheckpoint(tr_p + '_model.h5', period=1)

        epochs = int(args['epochs'])
        bs = int(args['batch_size'])

        if args['use_generator'] == 'True':

            ind_tr = get_ds(args['train_file'])
            train_gen = MissGen(ind_tr, bs=bs, chro=gn,
                                path='../bedGraph/chr_out/GSE63874_Na_' + stab + '_train.bedGraph',
                                stat='train', flank=flank)

            print('Starting training process')
            history = model.fit_generator(generator=train_gen, use_multiprocessing=True,
                                          max_queue_size=int(args['queue']), workers=int(args['workers']),
                                          steps_per_epoch=train_gen.__len__(), shuffle=False,
                                          epochs=epochs, verbose=1, callbacks=[mc])


        elif args['use_generator'] == 'False':

            if tr_ext == '.bedGraph':
                df = pd.read_csv(args['train_file'], header=None, names=["chr", "start", "end", "mm", "mp"], sep='\t')
                X = df.apply(lambda x: read_seq(x, gn, flank), axis=1)
                y = df['mm'].to_numpy()

            elif tr_ext == '.fa':
                df = pd.read_csv(args['train_file'], header=None)[0]
                X = df[1::2].str.upper()

                if not args['scores']:
                    print('use -sc to provide the path to the scores')
                    sys.exit()

                y = pd.read_csv(args['scores'], header=None)[0].to_numpy()

            l = X.apply(len)
            max_seq = np.max(l)
            X_oh = X.apply(lambda x: oneHot(x, max_seq))
            X_oh = X_oh.to_numpy()
            print('Starting training process')
            history = model.fit(x=X_oh, y=y, batch_size=bs, epochs=epochs, verbose=1, callbacks=[mc])

        print('Model was successfully trained!')
        out = open(tr_p + "/history2.pkl", "wb")
        pickle.dump(history.history, out)
        out.close()

        model.save(tr_p + '/model.h5')

    if args['use'] == 'test':

        '''
        this is a temporary version
        for the time being this script deals only with squences of length 15+flank*2
        all other sequences
        '''

        t_ext = os.path.splitext(args['test_file'])[1]

        if not t_ext == '.bedGraph' or not t_ext == '.fa':
            print('allowed input file extensions are bedGraph or fa')
            sys.exit()

        model = load_model(args['model_path'])

        if t_ext == '.bedGraph':
            df = pd.read_csv(args['test_file'], header=None, names=["chr", "start", "end", "mm", "mp"], sep='\t')
            l = list(np.arange(1, 23)) + ['X', 'Y', 'M']
            paths = list(map(lambda x: '../hg19/chr' + str(x) + '.fa', l))
            gn = Parallel(n_jobs=args['number_of_jobs'])(delayed(read_chr)(x) for x in paths)
            X = df.apply(lambda x: read_seq(x, gn, flank), axis=1)

        elif t_ext == '.fa':
            df = pd.read_csv(args['test_file'], header=None)[0]
            X = df[1::2].str.upper()


        l = X.apply(len)
        max_seq = np.max(model.input_shape[0][1])
        X = X[l <= max_seq] #temporary
        X_oh = X.apply(lambda x: oneHot(x, max_seq))
        X_oh = X_oh.to_numpy()
        print('Starting prediction process')
        pred = model.pedict(X_oh)

        res = pd.DataFrame({'sequence': X, 'scores': pred.squeeze()})
        res.to_csv(args['output_file'])
        print('All done!')


if __name__ == "__main__":
    main()














