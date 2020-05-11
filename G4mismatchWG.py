from utilsGGG import *
from models import model_base as mdl
import pickle
import sys
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd


def G4mismatchWG(args):

    flank = int(args['flank'])

    df = pd.read_csv(args['input'], header=None, sep='\t')
    if df.shape[1] == 1:
        ext = 'fa'
    elif df.shape[1] > 1:
        ext = 'bed'
        df.rename(columns={0: "chr", 1: "start", 2: "end"}, inplace=True)
        if df.shape[1] > 3:
            df.rename(columns={3: "score"}, inplace=True)

    if ext == 'bed':
        if not args['genome_path']:
            print('use -g to provide path to relevant genome assembly')
            sys.exit()
        else:
            gn = read_chr(args['genome_path'])

    if args['use'] == 'train':

        model = mdl((15 + (flank * 2), 4), filter_size=256, lr=1e-3, fc=32)
        mc = ModelCheckpoint(args['output'] + '_model.h5', period=1)
        epochs = int(args['epochs'])
        bs = int(args['batch_size'])

        if args['use_generator'] == 'True':
            if ext != 'bed':
                print('for now, the generator option may only be used for bed files')
                sys.exit()

            train_gen = MissGen(len(df), bs=bs, chro=gn, locs=df, stat='train', flank=flank)
            print('Starting training process')
            history = model.fit_generator(generator=train_gen, use_multiprocessing=True,
                                          max_queue_size=int(args['queue']), workers=int(args['workers']),
                                          steps_per_epoch=train_gen.__len__(), shuffle=False,
                                          epochs=epochs, verbose=1, callbacks=[mc])

        elif args['use_generator'] == 'False':
            if ext == 'bed':
                X = df.apply(lambda x: read_seq(x, gn, flank), axis=1)
                y = df['mm'].to_numpy()
            elif ext == 'fa':
                X = df[0][1::2].str.upper()
                if not args['scores']:
                    print('use -sc to provide target scores for training')
                    sys.exit()
                y = pd.read_csv(args['scores'], header=None)[0]

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

        if args['other_model']:
            p = args['other_model']
        else:
            p = 'WGmodels/' + args['stabilizer'] + '/model_' + str(flank) + '.h5'
        model = load_model(p)

        if ext == 'bed':
            X = df.apply(lambda x: read_seq(x, gn, flank), axis=1)

        elif ext == 'fa':
            X = df[1::2].str.upper()

        l = X.apply(len)
        max_seq = model.input_shape[0][1]
        X = X[l <= max_seq] #temporary
        X_oh = X.apply(lambda x: oneHot(x, max_seq))
        X_oh = X_oh.to_numpy()
        print('Starting prediction process')
        pred = model.pedict(X_oh)

        res = pd.DataFrame({'sequence': X, 'scores': pred.squeeze()})
        res.to_csv(args['output'] + 'G4mismatchWG_results.csv')
        print('All done!')

    return
