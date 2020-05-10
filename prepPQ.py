import pandas as pd
from joblib import Parallel, delayed
import argparse
import numpy as np
import sys
import os
from pq_utils import findPQ, feat_loop


def user_input():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_file', help='bed or fasta file for PQ detection', type=str, required=False)

    parser.add_argument('-g', '--generate', help='If set to True, the code will generate PQ squences based on the '
                                                 'desired genome,\n otherwise the code will just process the input '
                                                 'sequences.\n When set to true, an input of type bed is expected.',
                        choices=['True', 'False'], type=str, required=False, default='True')

    parser.add_argument('-f', '--flank', help='Sequence padding flank lengths.\n '
                                              'We used the 0, 50,100 and 150, but you may choose otherwise',
                        type=str, required=False, default='100')

    parser.add_argument('-r', '--regular_expression', help='Regular expression to look for in the data.',
                        type=str, required=False, default='(G{3,}[ACGTN]{1,12}){3,}G{3,}')

    parser.add_argument('-ml', '--min_loops', help='Minimal number of loops\n'
                                                   '(make sure it is consistent with the selected regular expression).',
                        type=str, required=False, default= '3')

    parser.add_argument('-ff', '--filter_flanks', help='If set to True sequences containing PQs in the '
                                                       'flanks will be filtered out.',
                        type=str, required=False, default='True')

    parser.add_argument('-gen', '--genome_path', help='Path to the folder containing the genome assembely of'
                                                      ' the desired organism.', type=str, required=False)

    parser.add_argument('-nj', '--number_of_jobs', help='Number of parallel threads for executing PQ search in '
                                                        'the input dataset',
                        type=str, required=False, default='1')

    parser.add_argument('-o', '--output_file', help='Name the output file', type=str, required=False, default='pqf')
    parser.add_argument('-sc', '--scores', help='Path to true score file for fasta input.',
                        type=str, required=False)

    args = parser.parse_args()
    arguments = vars(args)

    return arguments



args = user_input()

f = int(args['flank'])

if not args['input']:
    print('use -i to provide a path to an input file')
    sys.exit()

df = pd.read_csv(args['input'], header=None, sep='\t')
if df.shape[1] == 1:
    ext = 'fa'
elif df.shape[1] > 1:
    ext = 'bed'

if f == 0:
    args['filter_flanks'] == 'False'

if ext == 'bed':

    if not args['genome_path']:
        print('please provide the path to the folder containing the genome assembly')
        sys.exit()

    df.rename(columns={0: "chr", 1: "start", 2: "end"}, inplace=True)
    if df.shape[1] > 3:
        df.rename(columns={3: "score"}, inplace=True)

    u = df['chr'].unique()

    pq = Parallel(n_jobs=int(args['number_of_jobs']))(delayed(findPQ)(x, args, df[df['chr'] == x]) for x in u)

elif ext == 'fa':

    df = df[0][1::2]
    df = pd.DataFrame({'seq': df})
    if args['scores']:
        score = pd.read_csv(args['scores'], header=None)
        df['score'] = score[0]
    df = df[df['seq'].str.contains(args['regular_expression'])]
    df.reset_index(inplace=True, drop=True)
    df_match = df['seq'].apply(lambda x: list(args['regular_expression'].finditer(x)))
    df = df[df_match.apply(len) == 1]  # drop non-pq or pq in flanks
    df.reset_index(inplace=True, drop=True)
    df_match = df_match[df_match.apply(len) == 1]
    df_match.reset_index(inplace=True, drop=True)
    df_match = df_match.apply(lambda x: x[0].span())
    spans = pd.DataFrame(df_match.tolist(), index=df.index, columns=['s', 'e'])
    df = df[(spans['s'] <= f) & (df.apply(len) - spans['e'] <= f)] #drop long flanks
    df.reset_index(inplace=True, drop=True)
    df = pd.concat([df, spans], axis=1)
    pq = df.apply(lambda x: x[0][x['s']:x['e']], axis=1)
    df['loop'] = pq.apply(feat_loop, args=(int(args['min_loops']),))
    df.dropna(inplace=True) #not enough loops
    df.reset_index(inplace=True, drop=True)
    df['seq'] = df.apply(lambda x: (f - x['s'])*'N' + x[0] + 'N'*(f - len(x[0]) + x['e']), axis=1)

if args['output_file']:
    p = args['output_file']
else:
    p = 'G4detectorPQ_data.csv'

df_fin = pd.DataFrame({'seq': df['seq'], 'loop': df['loop']})
if np.isin('score', df.columns):
    df_fin['mm'] = df['score']

df_fin.to_csv(p, index=False)


















