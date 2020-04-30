from G4mismatchWG import G4mismatchWG
from G4mismatchPQ import G4mismatchPQ
import argparse
import sys





def user_input():

    parser = argparse.ArgumentParser()

    parser.add_argument('-gm', '--g4mm_model', help='Select G4mismatch model', choices=['WG', 'PQ'],
                        type=str, required=False)

    parser.add_argument('-u', '--use', help='Test with provided models, or train a new one.\n'
                                            'G4mismatchPQ can also be used for k-folds cross validation.',
                        choices=['train', 'test', 'cv'], type=str, required=False)

    parser.add_argument('-i', '--input', help='Path to input file.\n G4mismatchWG accepts bedGraph or FASTA files\n '
                                              'G4mismatchPQ accepts csv file generated with prep_pq.py',
                        type=str, required=False)

    parser.add_argument('-mf', '--model_feat', help='Input features for the PQ model',
                        choices=['base', 'split', 'split_numloop', 'split_looplen'],
                        type=str, required=False, default='split_numloop')

    parser.add_argument('-s', '--stabilizer', help='Stabilizing molecule.\n '
                                                   'We used K and PDS, but you may choose otherwise.',
                        choices=['K', 'PDS'], type=str, required=False, default='K')

    parser.add_argument('-f', '--flank', help='Sequence padding flank lengths.\n '
                                              'We used the 0, 50,100 and 150, but you may choose otherwise',
                        type=str, required=False, default='100')

    parser.add_argument('-o', '--output', help='Path to output folder.', type=str,
                        required=False)

    parser.add_argument('-g', '--genome_path', help='Path to the genome assembly of interest for bedGraph input',
                        type=str, required=False)

    parser.add_argument('-nj', '--number_of_jobs', help='Number of jobs for reading genome assembly',
                        type=str, required=False, default='1')

    parser.add_argument('-e', '--epochs', help='Number of training epochs', type=str, required=False, default='50')

    parser.add_argument('-bs', '--batch_size', help='Size of training batch', type=str, required=False, default='1000')

    parser.add_argument('-ug', '--use_generator', help='Boolean indicting if training is to be done with a generator.'
                                                       '\nAvailable for G4mismatchWG training.',
                        choices=['True', 'False'], type=str, required=False, default='False')

    parser.add_argument('-w', '--workers', help='Maximum number of processes to spin for generator',
                        type=str, required=False, default='40')

    parser.add_argument('-q', '--queue', help='Maximum queue size for generator',
                        type=str, required=False, default='100')

    parser.add_argument('-sc', '--scores', help='Target scores for fasta input in training mode oh G4mismatchWG',
                        type=str, required=False)

    parser.add_argument('-om', '--other_model', help='Path to model different from those provided.',
                        type=str, required=False)

    parser.add_argument('-fn', '--fold_num', help='Number of folds for k-fold cross-validation.',
                        type=str, required=False, default='3')

    parser.add_argument('-gh', '--get_history', help='If model training history is required, provide path to folder.',
                        type=str, required=False)

    parser.add_argument('-gcp', '--get_cv_preds', help='If prediction scores of cross-validation are required, '
                                                       'provide path to folder.', type=str, required=False)



    args = parser.parse_args()
    arguments = vars(args)

    return arguments



def main():

    args = user_input()

    if not args['input']:
        print('use -i to provide a path to an input file')
        sys.exit()

    if args['g4mm_model'] != 'WG' or args['g4mm_model'] != 'PQ':
        print('use -gm and select either PQ or WG method')
        sys.exit()

    if args['g4mm_model'] != 'WG' or args['g4mm_model'] != 'PQ':
        print('use -u to select if you want to use G4mismatch for train, cross-validation or test')
        sys.exit()

    else:
        if args['get_cv_preds'][-1] != '/':
            args['get_cv_preds'] = args['get_cv_preds'] + '/'

        if args['get_history'][-1] != '/':
            args['get_history'] = args['get_history'] + '/'

        if not args['output']:
            args['output'] = '/'
        elif args['output'][-1] != '/':
            args['output'] = args['output'] + '/'

        if args['g4mm_model'] == 'g4mmWG' and args['use'] == 'cv':
            print('WG method is only used for trein or test.')
            sys.exit()

        if args['g4mm_model'] == 'PQ':
           G4mismatchPQ(args)
        else:
            G4mismatchWG(args)





if __name__ == "__main__":
    main()
