import numpy as np
import pandas as pd
from mm_utils import oneHot, read_genome
from models import pearson
from mm_gen import GenScan
from tensorflow.keras.models import load_model
import argparse
import re
import tensorflow.keras.backend as K
import tensorflow as tf


def user_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--flank_size', help='Number of nt\'s to pad the sequence on each side',
                        type=int, default=100, required=False)
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int, required=False)
    parser.add_argument('-dp', '--data_path', help='Path to chromosome', type=str, required=False)
    parser.add_argument('-of', '--out_file', help='Path for the outputs', type=str,
                        required=False, default='.')
    parser.add_argument('-mp', '--model_path', help='Path to pre-trained model.', type=str, required=False)
    parser.add_argument('-rc', '--reverse_complement', help='Use reverse complement of chromosome.', type=int,
                        required=False, default=0, choices=[0, 1])
    parser.add_argument('-c', '--cut', help='Cut N tracts at the ends (for chromosome scanning).', type=int,
                        required=False, default=0, choices=[0, 1])
    parser.add_argument('-obg', '--out_bg', help='Name of output bedGraph file', type=str, required=False,
                        default='scan_res')
    parser.add_argument('-a', '--append',
                        help='If 1 append every sequence with flanks of Ns of length \"flank\", if 0 leave as is. '
                             '\nIf not used, the first and last 100 nt are considered flanks.',
                        type=int, required=False, default=0)
    parser.add_argument('-w', '--workers', help='Number of workers for batch generation.', type=int,
                        required=False, default=10)
    parser.add_argument('-fb', '--filter_bins',
                        help='If passed, bins with predicted mm score under the input value would be filtered out.',
                        type=float, required=False)
    parser.add_argument('-m', '--merge', help='Merge consecutive bins.',
                        type=int, required=False, default=0, choices=[0, 1])
    parser.add_argument('-mqs', '--max_queue_size', help='Max queu size.', type=int,
                        required=False, default=20)
    parser.add_argument('-mpr', '--multiprocessing', help='Use multiprocessing.', type=int,
                        required=False, default=1, choices=[0, 1])
    parser.add_argument('-vn', '--val_n',
                        help='Encoding of base N.',
                        type=float, required=False)

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

    # tf.config.set_visible_devices([], 'GPU')

    win_size = 15
    flank = args['flank_size']

    model = load_model(args['model_path'], custom_objects={"pearson": pearson}, compile=False)

    seqs = read_genome(args['data_path'])

    df = pd.DataFrame(columns=['name', 'start', 'end', 'pred'])

    s = 0

    for seq in seqs:
        # remove large N strands from both ends, but leave flanked windows on each side
        c = str(seq.seq)
        if args['append']:
            c = 'N' * flank + c + 'N' * flank
        if args['cut']:
            try:
                s = re.search(r'NNNNNN[ACGT]', c).end() - 1 - flank
            except:
                s = 0
            try:
                e = len(c) - re.search(r'NNNNNN[ACGT]', c[::-1]).end() + flank + 1
            except:
                e = len(c)
            print('cutting chromosome {} ends at {:d} and {:d}'.format(seq.name, s, e))
            c = c[s:e]

        c = oneHot(c, val_N=args['val_n'])

        if args['batch_size']:
            bs = args['batch_size']
        else:
            bs = len(c) - flank * 2 - win_size

        scan_gen = GenScan(win_size, bs=bs, seq=c, flank=flank, rc=args['reverse_complement'])

        pred = model.predict(scan_gen, use_multiprocessing=bool(args['multiprocessing']), workers=args['workers'],
                             max_queue_size=args['max_queue_size'])
        pred = pred.squeeze()
        num_win = len(pred)

        if args['merge']:
            if args['filter_bins'] is not None:
                tmp_df = pd.DataFrame(
                    {'name': [seq.name] * num_win,
                     'start': s + flank + np.arange(num_win),
                     'end': s + flank + np.arange(num_win) + win_size,
                     'pred': pred})
                tmp_df = tmp_df[tmp_df['pred'] >= args['filter_bins']]
                tmp_df.reset_index(inplace=True, drop=True)
                split_co = tmp_df['end'][:-1].reset_index(drop=True) - tmp_df['start'][1:].reset_index(drop=True)
                split_co[split_co > -1] = 0
                split_co = pd.concat([pd.Series([0]), split_co]).reset_index(drop=True)
                split = np.split(tmp_df, np.flatnonzero(split_co))
                tmp_df = pd.DataFrame(columns=['name', 'start', 'end', 'pred'])
                col_names = list(tmp_df)
                data = []
                for x in split:
                    l = [x['name'].iloc[0], x['start'].iloc[0], x['end'].iloc[-1], x['pred'].max()]
                    d = dict(zip(col_names, l))
                    data.append(d)
                tmp_df = tmp_df.append(data, True)
            else:
                tmp_df = pd.DataFrame(
                    {'name': [seq.name],
                     'start': [s + flank],
                     'end': [s + flank + num_win + win_size],
                     'pred': [pred.max()]})

        else:
            tmp_df = pd.DataFrame(
                {'name': [seq.name] * num_win,
                 'start': s + np.arange(num_win),
                 'end': s + np.arange(num_win) + win_size,
                 'pred': [pred]})

        df = pd.concat([df, tmp_df])

        K.clear_session()

    df.to_csv("{}/{}.bedGraph".format(args['out_file'], args['out_bg']),
              index=False, header=False, sep='\t', float_format='%.1f')


if __name__ == "__main__":
    args = user_input()
    main(args)
