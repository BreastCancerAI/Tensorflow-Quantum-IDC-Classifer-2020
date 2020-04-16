############################################################################################
#
# Project:       Breast Cancer AI Research Project
# Repository:    Tensorflow Quantum IDC Classifier 2020
# Project:       Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         IdcQnn
# Description:   IdcQnn is a wrapper class that creates the Invasive Ductal Carcinoma
#                Tensorflow QNN (Quantum Neural Network).
# License:       MIT License
# Last Modified: 2020-04-16
#
############################################################################################

import sys

from Classes.Helpers import Helpers
from Classes.Data import Data
from Classes.QMNIST import QMNIST
from Classes.QModel import QModel

class IdcQnn():
    """ IdcQnn

    IdcQnn is a wrapper class that creates the Invasive Ductal Carcinoma
    Tensorflow QNN (Quantum Neural Network).
    """

    def __init__(self):
        """ Initializes the class. """

        self.Helpers = Helpers("Core")
        self.QModel = QModel()

        self.Helpers.logger.info("IdcQnn QNN initialization complete.")

    def do_data(self):
        """ Sorts the training data """

        self.Data = Data()
        self.Data.get_paths_n_labels()
        self.Data.process_data()

    def do_train(self):
        """ Creates & trains the QNN. 
        
        CREDIT: https://www.tensorflow.org/quantum/tutorials/mnist
        CREDIT: https://arxiv.org/pdf/1802.06002.pdf 
        """

        self.QMNIST = QMNIST()

        # "Quantumize" the training data
        X_train_bin, X_test_bin = self.QMNIST.encode_data_as_binary(self.Data.X_train, self.Data.X_test)
        X_train_circ, X_test_circ = self.QMNIST.do_circuit_conversion(X_train_bin, X_test_bin)
        x_train_tfcirc, x_test_tfcirc = self.QMNIST.convert_to_tensors(X_train_circ, X_test_circ)
        
        # Create the Quantum Neural Network
        model_circuit, model_readout = self.QModel.create_quantum_model()
        self.QModel.create_keras_model(model_circuit, model_readout)
        
        # Train the Quantum Neural Network
        self.QModel.train_model(x_train_tfcirc, x_test_tfcirc, 
                                self.Data.y_train, self.Data.y_test)
        self.QModel.do_evaluate(x_train_tfcirc, x_test_tfcirc, self.Data.y_test)
        
IdcQnn = IdcQnn()

def main():

    if len(sys.argv) < 2:
        print("You must provide an argument")
        exit()
    elif sys.argv[1] not in IdcQnn.Helpers.confs["qnn"]["params"]:
        print("Mode not supported! Train or Classify")
        exit()

    mode = sys.argv[1]

    if mode == "Train":
        """ Creates and trains the classifier """
        IdcQnn.do_data()
        IdcQnn.do_train()
    else:
        """ Incorrect argument."""
        exit()

if __name__ == "__main__":
    main()