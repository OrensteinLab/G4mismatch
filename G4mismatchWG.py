from utilsGGG import *
from models import model_base as mdl
import pickle
import sys
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def G4mismatchWG(args):

    stab = args['stabilizer']
    flank = int(args['flank'])


    if args['use'] == 'train':

        tr_ext = os.path.splitext(args['input'])[1]

        if not tr_ext == '.bedGraph' or not tr_ext == '.fa':
            print('allowed input file extensions are bedGraph or fa')
            sys.exit()

        if tr_ext == '.bedGraph':
            if not args['genome_path']:
                print('use -g to provide path to relevant genome assembly')
                sys.exit()
            else:
                l = list(np.arange(1, 23)) + ['X', 'Y', 'M']
                paths = list(map(lambda x: args['genome_path'], l))
                gn = Parallel(n_jobs=args['number_of_jobs'])(delayed(read_chr)(x) for x in paths)

        model = mdl((15 + (flank * 2), 4), filters=256, lr=1e-3, fc=32)
        mc = ModelCheckpoint(args['output'] + '_model.h5', period=1)

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
                X.reset_index(inplace=True, drop=True)

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
        if args['get_history']:
            out = open(args['get_history'] + "history.pkl", "wb")
            pickle.dump(history.history, out)
            out.close()

        model.save(args['output'] + 'model.h5')

    elif args['use'] == 'test':

        '''
        this is a temporary version
        for the time being this script deals only with squences of length 15+flank*2
        all other sequences
        '''

        t_ext = os.path.splitext(args['input'])[1]

        if not t_ext == '.bedGraph' or not t_ext == '.fa':
            print('allowed input file extensions are bedGraph or fa')
            sys.exit()

        if args['other_model']:
            p = args['other_model']
        else:
            p = 'models/' + args['stabilizer'] + '/model_' + str(flank) + '.h5'
        model = load_model(p)

        if t_ext == '.bedGraph':
            df = pd.read_csv(args['input'], header=None, names=["chr", "start", "end", "mm", "mp"], sep='\t')
            l = list(np.arange(1, 23)) + ['X', 'Y', 'M']
            paths = list(map(lambda x: args['genome_path'], l))
            gn = Parallel(n_jobs=args['number_of_jobs'])(delayed(read_chr)(x) for x in paths)
            X = df.apply(lambda x: read_seq(x, gn, flank), axis=1)

        elif t_ext == '.fa':
            df = pd.read_csv(args['input'], header=None)[0]
            X = df[1::2].str.upper()

        l = X.apply(len)
        max_seq = np.max(model.input_shape[0][1])
        X = X[l <= max_seq] #temporary
        X_oh = X.apply(lambda x: oneHot(x, max_seq))
        X_oh = X_oh.to_numpy()
        print('Starting prediction process')
        pred = model.pedict(X_oh)

        res = pd.DataFrame({'sequence': X, 'scores': pred.squeeze()})
        res.to_csv(args['output'] + 'G4mismatchWG_results.csv')
        print('All done!')

    return















