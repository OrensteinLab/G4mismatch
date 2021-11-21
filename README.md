# G4mismatch

We present G4mismatch, a convolutional neural network for the prediction of DNA G-quadruplex (G4) mismatch scores. We couple Gemismatch with a scanner, capable of detecting potential G4forming sequences in any given input sequence.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The models supplied models were implemented using TensorFlow 2.3.

### Usage

To run G4mismast from command line, first change into its derectory.
G4mismatch requires several input arguments:
```
python G4mismatch.py \
       -tp <path to training set coordinate file> \
       -op <path for output directory> \
       -gp <path to relevent reference genome> \
```
Use `python G4mismatch.py --help` to view the complete list of input arguments.
The input to G4mismatch is a tab-deliminated file with 5 columns (no headers): chromosome, start, end, mismatch score and strand (`-` for forward strand, `+` for reverse strand, according to the G4-seq convention). An example of how the G4-seq human data is prepared for trainin is given in `prep_data.sh`

To use G4mismatch trained model for the detection of potential G4 forming sequences, run:
```
python G4mismatch_scan.py \
       -dp <path to fasta file> \
       -mp <path pre-trained G4mismatch model> \
       -of <path to directory for output files> \
       -m 1 \
       -fb 14.4 \
       -bs 128 \
```
To get the mismatch score for a full sequence the drop `-fb` argument.

<!--
#### Arguments
| Argument | Short hand | Description|
| ------------- | ------------- | -------------- |
| --g4mm_model | -gm | Denotes the G4mismatch method you would like to explore: <br> <li>`WG` - for whole genome models </li> <br> <li>`PQ` - for PQ models</li> |
| --use | -u |Denotes your use:<br> <li>`train` - for training a new model </li> <br> <li>`test` - for testing data with existing models</li> <br> <li>`cv` - k-fold cross validation (available for `PQ`) </li> |
| --input | -i  | Path to the input file you want to process. <br> `WG` accepts bed files with an additional fifth binary column, where 0 indicates forward strand and 1 is the forward strand, and fasta files.<br>`PQ` accepts csv files generated with `prepPQ.py` (coming soon). The pq_sample folder contains csv files with of PQs with 50nt flanks on each side generated from the G4-seq dataset.|
| --model_feat | -mf  | G4mismatch-PQ, has several stuctures, that differ from each othe in the way they accept the input. This argument denotes whis of these models you would like to explore.<br> <li>`base` - raw input sequence with concatinated flanks on each side. </li> <br> <li>`split` - the sequence and its flanks are used as separate inputs to the model</li> <br> <li>`split_numloop` - `split` input with an additional integer indicating the number of loops</li><br> <li>`split_looplen` - `split` input with an additional vector length of each loop</li>|
| --stabilizer | -s |Denotes one of two stabilizers used to generate the data G4mismatch midels were trained on:<br> <li>`K` </li> <br> <li>`PDS`</li><br>Default choice is `K` .|
| --flank | -f |Denotes the length of the flanks on each side of the processed sequence. Originally the G4mismatch models were trained with flank sizes of 0, 50, 100 and 150, but you're free to choose any size you prefer. Note that existing models can not process sequences larger then their original input size, these sequences will be dropped. Smaller sequences will be padded with zeros.<br> Default value is 100|
| --genome_path | -g |If a bed file is provided, G4mismatch requires a path to the genome assembly in order to extract the sequences. Note that Gemismatch requires that all the chromosomes will be in the same fasta file, where each chromosome begins with the header: `>chr<chromosome name>`.|
| --epochs | -e |Number of training epochs. Default value is 50.|
| --batch_size | -bs |Size of training batch.|
| --use_generator | -ug |Boolean indicating if a generator is to be used for training whole-genome models. Reccomended for very large datasets and available only for bed input. Default is set to False|
| --workers | -w |Maximum number of processes to spin when the generator is used.|
| --queue | -q |Maximum queue size when genrator is used.|
| --scores | -sc |Path to sequence scores, required for fasta input in train mode. For each sequence in the fasta file, this file should contain one value per line.|
| --othe_model | -om | Path to a trained model for test mode. Proved this argument if you wish to use a model defferent from the ones already provided.|
| --fold_num | -fn | Number of folds to be processed in k-folds cross-validation.|
| --get_history | -gh | If you wish to obtain the training history, provide a path to the destination folder. The file will be saved there under `history.pkl`.|
| --get_cv_preds | -gcp | If you wish to obtain the predictions made for each fold in the cross-validation mode, provide a path to the destination folder. The files will be saved there under `pq_scores_<fold number>.pkl`.|

-->

#### Datasets

The G4-seq data use for G4mismatch training is available at: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110582

For training G4mismatch the human chromosome 2 was used for validation, chromosome 1 was a held-out ttest set and the rest were used for training.

<!--
| Argument | Short hand | Description|
| ------------- | ------------- | -------------- |
| --input | -i  | Path to the input file you want to process. Your input file may be a bed file with chrom, chromStart and chromEnd columns and an optional strand and (required for testing and cross validation) score column, or you may provide a fasta file with the sequences.|
| --generate | -g  | If set to `True` (default) the code file look for all the PQ sequences in the desired genome assemble, otherwise the code will generate just process the input sequences.|
| --flank | -f |Denotes the length of the flanks on each side of the processed sequence. Originally the G4mismatch models were trained with flank sizes of 0, 50, 100 and 150, but you're free to choose any size you prefer. Note that existing models can not process sequences larger then their original input size, these sequences will be dropped. Smaller sequences will be padded with zeros.<br> Default value is 50.|
| --regular_expression | -r |Denotes the regular expression representing the PQ sequences.<br> Default value is `(G{3,}[ACGTN]{1,12}){3,}G{3,}`.|
| --filter_flanks | -ff |If set to `True` (default) samples containing PQs in the flanking sequences will be filtered out. |
| --genome_path | -gen |Path to the desired genome assembly file. Required for working with bed files. Note that `prepPQ.py` requires that for all the chromosomes to be concentrated in one fasta file, where each chromosome begins with the header: `>chr<chromosome name>`.|
| --number_of_jobs | -nj | Number of parallel threads executing PQ search. Default is set to 1.|
| --output_file | -o  | Name of the output file.|
| --scores | -sc |Path to sequence scores for fasta input. For each sequence in the fasta file, this file should contain one value per line.|

-->
<!--
### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
-->
