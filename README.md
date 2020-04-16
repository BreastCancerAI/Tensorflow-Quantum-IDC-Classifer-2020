# Breast Cancer AI Research Project
## Tensorflow Quantum IDC Classifier 2020
### IDC QNN, Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma

[![GeniSysAI Server](Media/Images/banner.png)](https://github.com/BreastCancerAI/Tensorflow-Quantum-IDC-Classifer-2020)

# Introduction
In this project we will leverage [Tensorflow Quantum](https://www.tensorflow.org/quantum "Tensorflow Quantum") [MNIST Classification](https://www.tensorflow.org/quantum/tutorials/mnist "MNIST Classification") code and modify the network to detect Invasive Ductal Carcinoma (IDC). This is an introductory tutorial that I made whilst learning the basics of Tensorflow Quantum for Quantum Neural Networks.

&nbsp;

# Hardware
I used the following hardware, but the tutorial should work on other NVIDIA GPUs.

- Intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8
- NVIDIA GTX 1050 Ti Ti/PCIe/SSE2

&nbsp;

# Operating system
- Ubuntu 18.04

&nbsp;

# Programming language
- Python 3.7

&nbsp;

# Software
In this project we have used the following core softwares:

- Tensorflow 2.1.0
- Tensorflow-Quantum

&nbsp;

# Tensorflow Quantum
"TensorFlow Quantum (TFQ) is a quantum machine learning library for rapid prototyping of hybrid quantum-classical ML models. Research in quantum algorithms and applications can leverage Google’s quantum computing frameworks, all from within TensorFlow.

TensorFlow Quantum focuses on quantum data and building hybrid quantum-classical models. It integrates quantum computing algorithms and logic designed in Cirq, and provides quantum computing primitives compatible with existing TensorFlow APIs, along with high-performance quantum circuit simulators. Read more in the TensorFlow Quantum white paper." [Source](https://www.tensorflow.org/quantum "Source")

&nbsp;

# Breast Histopathology Images
The dataset used in this project is an open dataset: [Breast Histopathology Images](https://www.kaggle.com/paultimothymooney/breast-histopathology-images "Breast Histopathology Images") by [Paul Mooney](https://www.kaggle.com/paultimothymooney "Paul Mooney") on [Kaggle](https://www.kaggle.com "Kaggle").

"The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x. From that, 277,524 patches of size 50 x 50 were extracted (198,738 IDC negative and 78,786 IDC positive). Each patch’s file name is of the format: uxXyYclassC.png — > example 10253idx5x1351y1101class0.png . Where u is the patient ID (10253idx5), X is the x-coordinate of where this patch was cropped from, Y is the y-coordinate of where this patch was cropped from, and C indicates the class where 0 is non-IDC and 1 is IDC." [Source](https://www.kaggle.com/paultimothymooney/breast-histopathology-images "Source")

In our project we will use a dataset made up from the Breast Histopathology Images.

&nbsp;

# Installation
Please follow the [Tensorflow Quantum IDC Classifier 2020 Installation Guide](Documentation/Installation/Installation.md "Tensorflow Quantum IDC Classifier 2020 Installation Guide") to install Tensorflow Quantum IDC Classifier 2020.

&nbsp;

# Training the Quantum Neural Network
Now you are ready to train your Quantum Neural Network. As mentioned above, an Ubuntu machine was used. Using different machines/GPU may vary the results, if so please let us know your findings.

## Start The Training
Ensuring you have completed all previous steps, you can start training using the following commands from the project root.

```
python3 IdcQnn.py Train
```

This tells the classifier to start in Train mode which will start the model training process.

### Data
First the data will be prepared.

```
2020-04-16 05:16:34,873 - Data - INFO - Data Helper Class initialization complete.
2020-04-16 05:16:34,890 - Data - INFO - Data Paths: 10000
2020-04-16 05:16:47,459 - Data - INFO - Data shuffled
2020-04-16 05:17:26,171 - Data - INFO - Converted data shape: (10000, 4, 4, 1)
2020-04-16 05:17:26,172 - Data - INFO - Encoded labels shape: (10000,)
2020-04-16 05:17:26,173 - Data - INFO - Training data: (7450, 4, 4, 1, 1)
2020-04-16 05:17:26,173 - Data - INFO - Training labels: (7450,)
2020-04-16 05:17:26,173 - Data - INFO - Validation data: (2550, 4, 4, 1, 1)
2020-04-16 05:17:26,173 - Data - INFO - Validation labels: (2550,)
```

You can find the code for this part of the tutorial in the [Classes/Data.py](Classes/Data.py "Classes/Data.py") file.

#### Start adding some of that Quantumness
Now we are starting to get to the interesting part! It is time to introduce some Quantum magic! In the Classification with Quantum Neural Networks on Near Term Processors paper, Farhi et al proposed that each pixel would be represented by a Qubit.

You can find the code for this part of the tutorial in the [Classes/QMNIST.py](Classes/QMNIST.py "Classes/QMNIST.py") file.

First the data is converted to a binary encoding, then each pixel in each image is converted into a Qubit, then finally we create Circ Circuits and convert them to Tensorflow Quantum Tensors.

```
2020-04-16 05:17:26,173 - QMNIST - INFO - QMNIST Helper Class initialization complete.
2020-04-16 05:17:26,174 - QMNIST - INFO - Data converted to binary encoding!
2020-04-16 05:17:32,850 - QMNIST - INFO - Data pixels converted to Qubits!
2020-04-16 05:17:42,309 - QMNIST - INFO - Converted Cirq circuits to TFQ tensors!
2020-04-16 05:17:42,317 - QModel - INFO - QNN model created.
```

### The Quantum Neural Network
Next the code will create the Quantum Neural Network we will use to detect IDC.

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
pqc (PQC)                    (None, 1)                 32        
=================================================================
Total params: 32
Trainable params: 32
Non-trainable params: 0
_________________________________________________________________
```

### The training
The training will now begin, if you are using the provided configuration, your network will train for 3 epochs. When complete you should see the same, or similar to, the following output.

```
Train on 7450 samples, validate on 2550 samples
Epoch 1/3
7450/7450 [==============================] - 1513s 203ms/sample - loss: 0.9296 - hinge_accuracy: 0.5653 - val_loss: 0.8434 - val_hinge_accuracy: 0.5965
Epoch 2/3
7450/7450 [==============================] - 1512s 203ms/sample - loss: 0.8396 - hinge_accuracy: 0.5847 - val_loss: 0.8051 - val_hinge_accuracy: 0.6020
Epoch 3/3
7450/7450 [==============================] - 1433s 192ms/sample - loss: 0.8218 - hinge_accuracy: 0.5886 - val_loss: 0.7989 - val_hinge_accuracy: 0.6047
2550/2550 [==============================] - 16s 6ms/sample - loss: 0.7989 - hinge_accuracy: 0.6047
```

# Results
We can see that the hinge accuracy is not very good. Increasing the dataset with actual training images (Non augmented) decreased the hinge accuracy considerably. In the next update to this tutorial we will go deeper into how to increase the accuracy of this model.

&nbsp;

# Contributing

The [Breast Cancer AI Research Project](https://github.com/COVID-19-AI-Research-Project "Breast Cancer AI Research Project") encourages, and welcomes, code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Peter Moss Leukemia AI Research](https://www.leukemiaresearchassociation.ai "Peter Moss Leukemia AI Research") Founder & Intel Software Innovator, Sabadell, Spain

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues

We use the [repo issues](issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.