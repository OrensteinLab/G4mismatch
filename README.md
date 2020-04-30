# G4mismatch

We present G4mismatch, a convolutional neural network for the prediction of DNA G-quadruplex (G4) mismatch scores. We developed and trained G4mismatch on two tasks: prediction of G4 mismatch score of any given DNA sequence with th WG model, and prediction of G4 propensity score for putative quadruplexes with the PQ model.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The models supplied models were implemented using Keras 2.1.6 with Tensorflow backend.

### Usage

To run G4mismast from command line, first change into its derectory.
G4mismatch requires several input arguments:
```
python G4mismatch.py \
       -gm <Select method WG or PQ> \
       -i <input file> \
       -u <use for cross-validations (for PQ), train or test> \
```
Use `python G4mismatch.py --help` to view the complete list of input arguments.

#### Arguments
| Argument | Short hand | Description|
| ------------- | ------------- | -------------- |
| --g4mm_model | -gm | Denotes the G4mismatch method you would like to explore: <br> <li>`WG` - for whole genome models </li> <br> <li>`PQ` - for PQ models</li> |
| --use | -u |Denotes your use:<br> <li>`train` - for training a new model </li> <br> <li>`test` - for testing data with existing models</li> <br> <li>`cv` - k-fold cross validation (avalable for `PQ`) </li> |
| --input | -i  | Path to the input file you want to process. <br> `WG` accepts bedGraphe files with an additional fifth binary column, where 0 indicates forward strand and 1 is the forward strand, and fasta files.<br>`PQ` accepts csv files generated with `prep_pq.py` (coming soon).|
| --model_feat | -mf  | G4mismatch-PQ, has several stuctures, that differ from each othe in the way they accept the input. This argument denotes whis of these models you would like to explore.<br> <li>`base` - raw input sequence with concatinated flanks on each side. </li> <br> <li>`split` - the sequence and its flanks are used as separate inputs to the model</li> <br> <li>`split_numloop` - `split` input with an additional integer indicating the number of loops</li><br> <li>`split_looplen` - `split` input with an additional vector length of each loop</li>|
| --stabilizer | -s |Denotes one of two stabilizers used to generate the data G4mismatch midels were trained on:<br> <li>`K` </li> <br> <li>`PDS`</li><br>Default choice is `K` .|
| --flank | -f |Denotes the length of the flanks on each side of the processed sequence. Originally the G4mismatch models were trained with flank sizes of 0, 50, 100 and 150, but you're free to choose any size you prefer. Note that existing models can not process sequences larger then their original input size, these sequences will be dropped. Smaller sequences will be padded with zeros.<br> Default value is 100|
| --genome_path | -g |If a bedGraph file is provided, G4mismatch requires a path to the genome assembly in order to extract the sequences.|
| --number_of_jobs | -nj |Number of jobs if you want to parallaze the reading of the genome assembly. Recommended if the assembly is split between multiple files (like hg19, for example)|
| --epochs | -e |Number of training epochs. Default value is 50.|
| --batch_size | -bs |Size of training batch.|
| --use_generator | -ug |Boolean indicating if a generator is to be used for training whole-genome models. Reccomended for very large datasets and available only for bedGraph input. Default is set to False|
| --workers | -w |Maximum number of processes to spin when the generator is used.|
| --queue | -q |Maximum queue size when genrator is used.|
| --scores | -sc |Path to sequence scores, required for fasta input in train mode. For each sequence in the fasta file, this file should contain one value per line|
| --othe_model | -om | Path to a trained model for test mode. Proved this argument if you wish to use a model defferent from the ones already provided.|
| --fold_num | -fn | Number of fods to be processed in k-folds cross-validation.|
| --get_history | -gh | If you wish to obtain the training history, provide a path to the destination folder. The file will be saved there under `history.pkl`.|
| --get_cv_preds | -gcp | If you wish to obtain the predictions made for each fold in the cross-validation mode, provide a path to the destination folder. The files will be saved there under `pq_scores_<fold number>.pkl`.|



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
