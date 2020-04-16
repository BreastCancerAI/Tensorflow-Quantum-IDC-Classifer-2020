############################################################################################
#
# Project:       Breast Cancer AI Research Project
# Repository:    Tensorflow Quantum IDC Classifier 2020
# Project:       Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Data Helper Class
# Description:   Data functions for the Leveraging Quantum MNIST to detect Invasive Ductal 
#                Carcinoma QNN (Quantum Neural Network).
# License:       MIT License
# Last Modified: 2020-04-16
#
############################################################################################

import collections, os, random

import numpy as np
import tensorflow as tf

from random import seed as rseed
from numpy.random import seed as nseed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle as skshuffle
import matplotlib.pyplot as plt

from Classes.Helpers import Helpers

class Data():
    """ Data Helper Class

    Data functions for the Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma 
    QNN (Quantum Neural Network).
    """

    def __init__(self):
        """ Initializes the class. """

        self.Helpers = Helpers("Data", False)

        self.dim = self.Helpers.confs["qnn"]["data"]["dim"]
        self.dir_train = self.Helpers.confs["qnn"]["data"]["dir_train"]
        self.seed = self.Helpers.confs["qnn"]["data"]["seed"]

        nseed(self.seed)
        rseed(self.seed)

        self.data = []
        self.labels = []
        self.paths = []

        self.Helpers.logger.info("Data Helper Class initialization complete.")

    def get_paths_n_labels(self):
        """ Stores data paths and labels as a list of tuples. """

        for ddir in os.listdir(self.dir_train):
            tpath = os.path.join(self.dir_train, ddir)
            if os.path.isdir(tpath):
                for filename in os.listdir(tpath):
                    if filename.lower().endswith(tuple(self.Helpers.confs["qnn"]["data"]["allowed"])):
                        self.paths.append((os.path.join(tpath, filename), ddir))
                    else:
                        continue

        self.Helpers.logger.info("Data Paths: " + str(len(self.paths)))

    def process_data(self):
        """ Processes the data. """

        for tdata in self.paths:
            (image, label) = (tdata[0], tdata[1])
            
            image_string = tf.io.read_file(image)
            image_decoded = tf.image.decode_png(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            image = tf.image.resize(image, [4, 4])
            image = tf.image.rgb_to_grayscale(image)

            self.data.append(image)
            self.labels.append(label == "1")

        self.shuffle()
        self.convert_data()
        self.encode_labels()
        self.get_split()

    def shuffle(self):
        """ Shuffles the data and labels. """

        self.data, self.labels = skshuffle(self.data, self.labels, random_state = self.seed)

        self.Helpers.logger.info("Data shuffled")

    def convert_data(self):
        """ Converts the training data to a numpy array. """

        self.data = np.array(self.data)

        self.Helpers.logger.info("Converted data shape: " + str(self.data.shape))

    def encode_labels(self):
        """ One Hot Encodes the labels. """
        
        self.labels = np.array(self.labels)

        self.Helpers.logger.info("Encoded labels shape: " + str(self.labels.shape))

    def get_split(self):
        """ Splits the data and labels creating training and validation datasets. """
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=0.255, random_state = self.seed)

        self.X_train, self.X_test = self.X_train[..., np.newaxis]/255.0, self.X_test[..., np.newaxis]/255.0
        
        self.Helpers.logger.info("Training data: " + str(self.X_train.shape))
        self.Helpers.logger.info("Training labels: " + str(self.y_train.shape))
        self.Helpers.logger.info("Validation data: " + str(self.X_test.shape))
        self.Helpers.logger.info("Validation labels: " + str(self.y_test.shape))