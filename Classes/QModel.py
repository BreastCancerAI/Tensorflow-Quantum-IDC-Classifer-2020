############################################################################################
#
# Project:       Breast Cancer AI Research Project
# Repository:    Tensorflow Quantum IDC Classifier 2020
# Project:       Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         QModel Helper Class
# Description:   Quntum Model helper functions for the Leveraging Quantum MNIST to detect  
#                Invasive Ductal Carcinoma QNN (Quantum Neural Network).
# Credit:        Tensorflow Quantum MNIST classification
#                https://www.tensorflow.org/quantum/tutorials/mnist
# Credit:        Classification with Quantum Neural Networks on Near Term Processors
#                https://arxiv.org/pdf/1802.06002.pdf
# License:       MIT License
# Last Modified: 2020-04-16
#
############################################################################################

import cirq, collections, sympy

import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

from Classes.Helpers import Helpers
        
class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

class QModel():
    """ QModel Helper Class

    Quantum Model helper functions for the Leveraging Quantum MNIST to detect 
    COVID-19 QNN (Quantum Neural Network).

    CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
    """

    def __init__(self):
        """ Initializes the class. """

        self.Helpers = Helpers("QModel", False)

        self.dim = self.Helpers.confs["qnn"]["data"]["dim"]

        self.Helpers.logger.info("QModel Helper Class initialization complete.")
        
    def create_quantum_model(self):
        """Create a QNN model circuit and readout operation to go along with it.

        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf
        """
        
        data_qubits = cirq.GridQubit.rect(self.dim, self.dim) # a 4x4 grid.
        readout = cirq.GridQubit(-1, -1) # a single qubit at [-1,-1]
        circuit = cirq.Circuit()
        
        # Prepare readout qubit.
        circuit.append(cirq.X(readout))
        circuit.append(cirq.H(readout))
        
        builder = CircuitLayerBuilder(
            data_qubits = data_qubits,
            readout=readout)

        # Add layers.
        builder.add_layer(circuit, cirq.XX, "xx1")
        builder.add_layer(circuit, cirq.ZZ, "zz1")

        # Finally, append the readout qubit.
        circuit.append(cirq.H(readout))

        self.Helpers.logger.info("QNN model created.")

        return circuit, cirq.Z(readout)

    def create_keras_model(self, model_circuit, model_readout):
        """ Create the QNN network.

        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf
        """
        
        self.model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the readout gate, range [-1,1].
            tfq.layers.PQC(model_circuit, model_readout),
        ])
        
        self.model.compile(loss=tf.keras.losses.Hinge(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[self.hinge_accuracy])
        
        print(self.model.summary())
        
    def hinge_accuracy(self, y_true, y_pred):
        """Create a QNN model circuit and readout operation to go along with it.

        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf
        """
        
        y_true = tf.squeeze(y_true) > 0.0
        y_pred = tf.squeeze(y_pred) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)

        return tf.reduce_mean(result)
        
    def train_model(self, x_train_tfcirc, x_test_tfcirc, y_train, y_test):
        """Create a QNN model circuit and readout operation to go along with it.

        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf
        """

        NUM_EXAMPLES = len(x_train_tfcirc)
        
        y_train_hinge = 2.0*y_train-1.0
        y_test_hinge = 2.0*y_test-1.0
        
        x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
        y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]
        
        self.qnn_history = self.model.fit(
            x_train_tfcirc_sub, y_train_hinge_sub,
            batch_size = self.Helpers.confs["qnn"]["train"]["batch_size"],
            epochs = self.Helpers.confs["qnn"]["train"]["epochs"],
            verbose = self.Helpers.confs["qnn"]["train"]["verbose"],
            validation_data=(x_test_tfcirc, y_test_hinge))

    def do_evaluate(self, x_train_tfcirc, x_test_tfcirc, y_test):
        """ Evaluates the model """

        qnn_results = self.model.evaluate(x_test_tfcirc, y_test)