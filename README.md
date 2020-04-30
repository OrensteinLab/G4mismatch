# G4mismatch

We present G4mismatch, a convolutional neural network for the prediction of DNA G-quadruplex (G4) mismatch scores. We developed and trained G4mismatch on two tasks: prediction of G4 mismatch score of any given DNA sequence, and prediction of G4 propensity score for putative quadruplexes.

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

## Arguments
| Argument | Short hand | Description|
| :-------------: | :-------------: |:--------------:|
| g4mm_model | gm  | Denotes the G4mismatch method you would like to explore:<br> <li>`WG` - for whole genome models </li> <br> <li>`PQ` - for PQ models</li> |
| ------------- | ------------- |--------------|
| use  | u  |Denotes your use :<br> <li>`train` - for training a new model </li> <br> <li>`test` - for testing data with existing models</li> <br> <li>`cv` - k-fold cross validation (avalable for `PQ`) </li> |


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
