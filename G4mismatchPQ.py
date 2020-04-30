from models import *
from scipy.stats.stats import pearsonr
from keras import backend as K
import pickle
from utilsGGG import *
from keras.models import load_model
from pq_utils import prep_data
import pandas as pd
import numpy as np



def G4mismatchPQ(args):

    stab = args['stabilizer']
    flank = int(args['flank'])

    df = pd.read_csv(args['input'])
    x = df['seq']
    loops = df['loop'].apply(loop_info)

    if args['model_feat'] != 'test':

        X_in = prep_data(x, args, loops=loops)
        y = df['mm'].to_numpy()

        if args['model_feat'] == 'train':
            fold_size = df.shape[0]
        else:
            fold_size = int(np.ceil(df.shape[0] / int(args['fold_num'])))

        folds = np.arange(0, df.shape[0], fold_size)

        num_epochs = args['epoch']
        bs = args['batch_size']
        c = []

        for i, start in enumerate(folds):

            if args['model_feat'] == 'cv':
                x_test, y_test, x_train, y_train = toFolds(fold_size, start, X_in, y)

            else:
                x_train, y_train = X_in, y

            if args['model_feat'] == 'base':

                model = model_base((x_train.shape[1], x_train.shape[2]), filter_size=256, lr=1e-3, fc=32)

            elif args['model_feat'] == 'split':
                model = model_split((x_train[0].shape[1], x_train[0].shape[2]),
                                    (x_train[1].shape[1], x_train[1].shape[2]), filter_size=256, lr=1e-3, fc=32)

            else:
                model = model_loop((x_train[0].shape[1], x_train[0].shape[2]),
                                   (x_train[1].shape[1], x_train[1].shape[2]),
                                   x_train[3].shape[0], filter_size=256, lr=1e-3, fc=32)

            history = model.fit(x=x_train, y=y_train, batch_size=bs, epochs=num_epochs, verbose=0)

            if args['use'] == 'cv':
                pred = model.predict(x_test)
                pred = pred.squeeze()
                c.append(pearsonr(y_test, pred)[0])
                K.clear_session()

                if args['get_cv_preds']:
                    if start + fold_size > len(df):
                        end = len(df)
                    else:
                        end = start + fold_size


                    df_scores = pd.DataFrame({"seq": x[start:end], "label": y_test, "score": pred})
                    df_scores.to_csv(args['get_cv_preds'] + "pq_scores_" + str(i) + ".csv")

            if args['get_history']:
                out = open(args['get_history'] + "history_" + str(i) + ".pkl", "wb")
                pickle.dump(history.history, out)
                out.close()

        if args['use'] == 'train':
            print('Model was successfully trained!')
            model.save(args['output'] + 'model.h5')

        else:
            mc = np.mean(c)
            print("Cross-validation is complete.\n Mean pearson correlation is: " + "{:.2f}".format(mc))

    elif args['use'] == 'test':

        if args['other_model']:
            p = args['other_model']
        else:
            p = 'models/' + args['model_config'] + '/' + stab + '/model_' + str(flank) + '.h5'
        model = load_model(p)
        max_seq = np.max(model.input_shape[0][1])
        l = x.apply(len)
        x = x[l <= max_seq]
        x.reset_index(inplace=True, drop=True)
        X_in = prep_data(x, args, p=max_seq, loops=loops)

        pred = model.fit(X_in)
        df_scores = pd.DataFrame({"seq": x, "score": pred})
        df_scores.to_csv(args['output']+'G4mismatchPQ_results.csv')

        print('Testing process is complete!')

    return

