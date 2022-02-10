""" Project constants. """

# Imports ---------------------------------------------------------------------

import os

# Constants -------------------------------------------------------------------

DATA_DIR = 'data'
TFDS_DIR = os.path.join(DATA_DIR, 'tfds')
DATASET_FILE= os.path.join(DATA_DIR, 'dataset', 'dataset.csv')
LOG_DIR = 'logs'
MODEL_DIR = 'models'
VECTORIZER_DIR = 'vectorizers'
PLOT_DIR = 'plots'
STYLE_FILE = 'project.mplstyle'
STYLE_PATH = os.path.join(PLOT_DIR, STYLE_FILE)
SEED = 271828
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 128