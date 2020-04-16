############################################################################################
#
# Project:       Breast Cancer AI Research Project
# Repository:    Tensorflow Quantum IDC Classifier 2020
# Project:       Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         QMNIST Helper Class
# Description:   Wrapper classed based on Tensorflow Quantum MNIST example.
# Credit:        Tensorflow Quantum MNIST classification
#                https://www.tensorflow.org/quantum/tutorials/mnist
# Credit:        Classification with Quantum Neural Networks on Near Term Processors
#                https://arxiv.org/pdf/1802.06002.pdf
# License:       MIT License
# Last Modified: 2020-04-16
#
############################################################################################

import cirq, sympy

import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np

from Classes.Helpers import Helpers

class QMNIST():
    """ QMNIST Helper Class

    QMNIST functions for the Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma 
    QNN (Quantum Neural Network).

    CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
    CREDIT: https://arxiv.org/pdf/1802.06002.pdf
    """

    def __init__(self):
        """ Initializes the class. """

        self.Helpers = Helpers("QMNIST", False)

        self.bthreshold = self.Helpers.confs["qnn"]["core"]["bin_threshold"]
        self.dim = self.Helpers.confs["qnn"]["data"]["dim"]

        self.Helpers.logger.info("QMNIST Helper Class initialization complete.")

    def encode_data_as_binary(self, x_train, x_test):
        """ Converts to a binary encoding.

        In the Classification with Quantum Neural Networks on Near Term
        Processors paper, Farhi et al proposed that each pixel would
        be represented by a Qubit, this requires the data to be first
        be binary encoded.

        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf
        """

        x_train_bin = np.array(x_train > self.bthreshold, dtype=np.float32)
        x_test_bin = np.array(x_test > self.bthreshold, dtype=np.float32)

        self.Helpers.logger.info("Data converted to binary encoding!")

        return x_train_bin, x_test_bin

    def convert_to_circuit(self, image):
        """ Encode truncated classical image into quantum datapoint.

        The qubits at pixel indices with values that exceed a threshold,
        are rotated through an X gate.

        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf
        """
        
        values = np.ndarray.flatten(image)
        qubits = cirq.GridQubit.rect(self.dim, self.dim)
        circuit = cirq.Circuit()
        for i, value in enumerate(values):
            if value:
                circuit.append(cirq.X(qubits[i]))
        return circuit

    def do_circuit_conversion(self, X_train_bin, X_test_bin):
        """ Encodes images as quantum data points.

        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf
        """

        X__train_circ = [self.convert_to_circuit(x) for x in X_train_bin]
        X__test_circ = [self.convert_to_circuit(x) for x in X_test_bin]

        self.Helpers.logger.info("Data pixels converted to Qubits!")

        return X__train_circ, X__test_circ

    def convert_to_tensors(self, x_train_circ, x_test_circ):
        """ Converts Cirq circuits to TFQ tensors.

        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf
        """

        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

        self.Helpers.logger.info("Converted Cirq circuits to TFQ tensors!")

        return x_train_tfcirc, x_test_tfcirc

