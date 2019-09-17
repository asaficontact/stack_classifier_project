
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pickle
import seaborn as sns
import requests
import string
import re


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer(sparse_output = True)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(lowercase = True, max_df = 0.95, stop_words='english')
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings('ignore')

import spacy
nlp = spacy.load("en_core_web_lg")

from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
import numpy as np
