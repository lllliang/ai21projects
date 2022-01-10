from __future__ import absolute_import, division, print_function

import sys
import random
import numpy as np
import torch
import pickle

def load_data(file_name):
    sentences = pickle.load(open(file_name,'rb'))
    return sentences

