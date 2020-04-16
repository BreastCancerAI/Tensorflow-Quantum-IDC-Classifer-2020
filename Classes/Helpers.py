############################################################################################
#
# Project:       Breast Cancer AI Research Project
# Repository:    Tensorflow Quantum IDC Classifier 2020
# Project:       Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Helper Class
# Description:   Helper functions for the Leveraging Quantum MNIST to detect Invasive Ductal 
#                Carcinoma QNN (Quantum Neural Network).
# License:       MIT License
# Last Modified: 2020-04-16
#
############################################################################################

import sys
import time
import logging
import logging.handlers as handlers
import json

from datetime import datetime


class Helpers():
    """ Helper Class
    
    Helper functions for the Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma 
    QNN (Quantum Neural Network).
    """

    def __init__(self, ltype, log=True):
        """ Initializes the Helpers Class. """

        self.confs = {}
        self.loadConfs()

        self.logger = logging.getLogger(ltype)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        allLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/all.log', when='H', interval=1, backupCount=0)
        allLogHandler.setLevel(logging.INFO)
        allLogHandler.setFormatter(formatter)

        errorLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/error.log', when='H', interval=1, backupCount=0)
        errorLogHandler.setLevel(logging.ERROR)
        errorLogHandler.setFormatter(formatter)

        warningLogHandler = handlers.TimedRotatingFileHandler(
            'Logs/warning.log', when='H', interval=1, backupCount=0)
        warningLogHandler.setLevel(logging.WARNING)
        warningLogHandler.setFormatter(formatter)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)

        self.logger.addHandler(allLogHandler)
        self.logger.addHandler(errorLogHandler)
        self.logger.addHandler(warningLogHandler)
        self.logger.addHandler(consoleHandler)

        if log is True:
            self.logger.info("Helpers class initialization complete.")

    def loadConfs(self):
        """ Load the program configuration. """

        with open('Model/config.json') as confs:
            self.confs = json.loads(confs.read())