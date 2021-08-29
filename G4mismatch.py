import pandas as pd
import tensorflow as tf
pd.options.mode.chained_assignment = None  # default='warn'
from mm_utils import *
from models import gen_model, pearson
from mm_gen import MissGen
from tensorflow.keras.models import load_model
import pickle
import argparse
from scipy.stats import pearsonr
import os
from tensorflow.keras.callbacks import ModelCheckpoint


def user_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--flank_size', help='Number of nt\'s to pad the sequence on each side',
                        type=int, default=50, required=False)
    parser.add_argument('-tp', '--train_path', help='Path to data', type=str, required=True)
    parser.add_argument('-gp', '--genome_path', help='Path to genome fasta file', type=str, required=False,
                        default='genomes/hg19/hg19.fa')
    parser.add_argument('-vp', '--val_path', help='Path to validation data', type=str, required=False)
    parser.add_argument('-op', '--out_path', help='Path for general diractory for saving outputs', type=str,
                        required=True)
    parser.add_argument('-ne', '--num_epochs', help='Number of training epochs', type=int, required=False, default=50)
    parser.add_argument('-bs', '--batch_size', help='Number of samples in a mini-batch', type=int, required=False,
                        default=1000)
    parser.add_argument('-fs', '--filter_size', nargs='+',
                        help='Size of 1D-conv filter, one integer argument for base model and two for split',
                        type=int, required=False, default=3)
    parser.add_argument('-mp', '--model_path', help='Path to pre-trained model. '
                                                    'If provided with \'train\' parameter, '
                                                    'the model will be fine-tuned, otherwise the model '
                                                    'is used for prediction', type=str, required=False)
    parser.add_argument('-vn', '--val_n', help='Encoding of base N.', type=float, required=False)
    parser.add_argument('-af', '--add_flank', help='Pass if a flank is to be added only of one side of the main bin.',
                        type=str, required=False, choices=['first', 'last'])
    parser.add_argument('-w', '--workers', help='Number of workers for batch generation.', type=int,
                        required=False, default=13)
    parser.add_argument('-mqs', '--max_queue_size', help='Max queu size.', type=int,
                        required=False, default=30)

    args = parser.parse_args()
    arguments = vars(args)

    return arguments


def main(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    flank = args['flank_size']
    w = 15
    filters = 256
    lr = 1e-3
    hidden = 32
    p = args['out_path']
    filter_size = args['filter_size']

    input_shape = (w + (flank * 2), 4)

    cb = [ModelCheckpoint(p + '/model{epoch:02d}.h5', save_freq='epoch', verbose=1)]

    if args['model_path']:
        model = load_model(args['model_path'], custom_objects={"pearson": pearson})  # load pre-trained model
    else:
        os.mkdir(p)

        model = gen_model(input_shape,
                          num_filters=filters,
                          filter_size=filter_size,
                          lr=lr,
                          fc=hidden)
    print(model.summary())
    train_locs = pd.read_csv(args['train_path'], header=None, names=["chr", "start", "end", "mm", "mp"], sep='\t')
    len_seq = train_locs['end'] - train_locs['start']
    rv_ind = len_seq[len_seq != w].index
    train_locs = train_locs.drop(index=rv_ind).reset_index(drop=True)
    chr_list = list(train_locs['chr'].unique())
    genome = read_genome(args['genome_path'], chr_list)
    train_gen = MissGen(bs=args['batch_size'],
                        genome=genome,
                        locs=train_locs,
                        stat='train',
                        flank=flank,
                        val_N=args['val_n'])

    if not args['val_path']:
        val_gen = None
        val_steps = None

    else:
        val_locs = pd.read_csv(args['val_path'], header=None, names=["chr", "start", "end", "mm", "mp"], sep='\t')
        len_seq = val_locs['end'] - val_locs['start']
        rv_ind = len_seq[len_seq != w].index
        val_locs = val_locs.drop(index=rv_ind).reset_index(drop=True)
        val_chr_list = list(val_locs['chr'].unique())
        val_genome = read_genome(args['genome_path'], val_chr_list)
        val_gen = MissGen(bs=args['batch_size'],
                          genome=val_genome,
                          locs=val_locs,
                          stat='train',
                          flank=flank,
                          val_N=args['val_n'])
        val_steps = int(val_gen.__len__())

    history = model.fit(x=train_gen,  validation_data=val_gen,
                        use_multiprocessing=True, max_queue_size=args['max_queue_size'], workers=args['workers'],
                        steps_per_epoch=int(train_gen.__len__()), validation_steps=val_steps, shuffle=False,
                        epochs=args['num_epochs'], callbacks=cb, verbose=2)

    out = open(p + "/history.pkl", "wb")
    pickle.dump(history.history, out)
    out.close()
    model.save(p + '/model.h5')

    if val_gen is not None:
        pred = model.predict(x=val_gen, use_multiprocessing=False, workers=0, max_queue_size=1, verbose=2)
        val_locs['score'] = pred.squeeze()
        val_locs.to_csv(p + '/results.csv', index=False)
        r = pearsonr(pred.squeeze(), np.array(val_locs['mm']))
        print('pearson correalation = {:.3f}, p value = {:.3f}'.format(r[0], r[1]))


if __name__ == "__main__":
    args = user_input()
    main(args)
