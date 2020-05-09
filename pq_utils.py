from utilsGGG import *


def feat_loop(seq, minl):
    loops = re.split('GGG+', seq)
    nloops = len(loops) - 2
    if nloops < minl:
        return None
    lenloops = list(map(len, loops[1:-1]))

    return lenloops


def get_score(seq, f, df_chr):

    df_chr = df_chr[(df_chr['start'] >= seq['start']) & (df_chr['end'] <= seq['end'])]
    if not len(df_chr):
        return None
    score = np.asarray(df_chr['mm']).max()

    return score


def findPQ(chr_n, args, df_chr=None):

    f = int(args['flank'])
    regexp = args['regular_expression']
    path = args['genome_path'] + chr_n + '.fa'
    chro = pd.read_csv(path)
    chro = chro['>' + chr_n].str.cat(sep='')
    chro = chro.upper()
    if args['generate'] == 'True':
        pq = list(regexp.finditer(chro))
        pq_spans = list(map(lambda x: x.span(), pq))
        seqs = pd.DataFrame(map(lambda x: chro[x[0] - f:x[1] + f], pq_spans), columns=['seq'])
        seqs['start'], seqs['end'] = zip(*pq_spans)
        seqs['start'] -= f
        seqs['end'] += f
        if np.isin('score', df_chr.columns):
            df_chr.reset_index(inplace=True, drop=True)
            seqs['score'] = seqs.apply(get_score, args=(f, df_chr), axis=1)
            seqs.dropna(inplace=True)
            seqs.reset_index(inplace=True, drop=True)

    else:
        df_chr.reset_index(inplace=True, drop=True)
        df_chr['seq'] = df_chr.apply(lambda x: chro[x['start'] - f:x['end'] + f], axis=1)
        seqs = df_chr
        del df_chr


    if args['filter_flanks'] == 'True':
        f1 = seqs['seq'].str[:f]
        drop1 = f1.str.contains(regexp.pattern)
        df1 = seqs[~drop1]
        df1.reset_index(inplace=True, drop=True)

        f2 = df1['seq'].str[-f:]
        drop2 = f2.str.contains(regexp.pattern)
        df2 = df1[~drop2]
        df2.reset_index(inplace=True, drop=True)
    else:
        df2 = seqs
    del seqs

    df2['loop'] = df2['seq'].str[f:-f].apply(feat_loop, args=(int(args['min_loops']),))
    df2.dropna(inplace=True)
    df2.reset_index(inplace=True, drop=True)
    df2['chr'] = chr_n

    return df2


def prep_data(data, args, p=None, loops=None):

    flank = args['flank']

    if args['model_feat'] == 'base':
        if p:
            max_seq = p
        else:
            max_seq = data.apply(len).max()

        X_oh = np.stack(data.apply(lambda x: oneHot(x, p)).values)
        X_in = [X_oh]

    else:
        X = data.str[flank:-flank]

        if p:
            max_seq = p
        else:
            max_seq = X.apply(len).max()

        f1 = data.str[:flank]
        f2 = data.str[-flank:]

        X_oh = np.stack(X.apply(lambda x: oneHot(x, max_seq)).values)
        f1_oh = np.stack(f1.apply(lambda x: oneHot(x)).values)
        f2_oh = np.stack(f2.apply(lambda x: oneHot(x)).values)

        X_in = [X_oh, f1_oh, f2_oh]

        if args['model_feat'] == 'split_numloops':
            loop_in = loops.apply(len)
            loop_in = np.expand_dims(loop_in.to_numpy(), 1)
            X_in.append(loop_in)

        elif args['model_feat'] == 'split_looplen':
            max_loops = max(loops.apply(len))
            loop_in = np.stack(loops.apply(
                lambda x: np.pad(x, (0, max_loops - len(x)), 'constant', constant_values=0)).values)
            X_in.append(loop_in)

    return X_in


