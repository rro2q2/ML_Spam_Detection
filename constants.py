import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from string import punctuation
import timeit
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
#nltk.download('stopwords')
#nltk.download('genesis')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
genesis_ic = wn.ic(genesis, False, 0.0)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

from metrics import plot_confusion_matrix

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

## Filenames ##
SPAM_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/spam.csv')

