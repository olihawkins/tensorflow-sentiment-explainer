"""
Prepare and load datasets.
"""

# Imports ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical

from sentiment.constants import DATA_DIR
from sentiment.constants import TFDS_DIR
from sentiment.constants import DATASET_FILE
from sentiment.constants import SEED

# Create a dataset from numpy arrays of documents and labels ------------------

def create_tf_dataset(
    documents, 
    labels, 
    batch_size=32,
    seed=None):

    dataset = Dataset.from_tensor_slices((documents, labels))
    dataset = dataset.shuffle(dataset.__len__(), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    return dataset

# Prepare dataset -------------------------------------------------------------

def prepare_dataset():

    # Download or load the tfds imdb dataset
    imdb, info = tfds.load(
        "imdb_reviews", 
        data_dir=TFDS_DIR,
        as_supervised=True, 
        with_info=True)

    # Combine train and test sets to repartition
    documents = []
    labels = []
    for document, label in tfds.as_numpy(imdb['train']):
        document = document.decode('utf-8').replace('<br />', '')
        documents.append(document)
        labels.append(label)

    for document, label in tfds.as_numpy(imdb['test']):
        document = document.decode('utf-8').replace('<br />', '')
        documents.append(document)
        labels.append(label)

    documents = np.array(documents)
    labels = np.array(labels)

    df = pd.DataFrame({
        'documents': documents,
        'labels': labels})

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.to_csv(DATASET_FILE, index=False)

# Prepare dataset -------------------------------------------------------------

def load_dataset(
    train_size=0.8,
    val_size=0.1,
    test_size=0.1):

    df = pd.read_csv(DATASET_FILE)

    # Split into train, validation and test datasets
    train_end = np.floor(train_size * df.shape[0]).astype(int)
    val_end = np.floor((train_size + val_size) * df.shape[0]).astype(int)
    
    train_df = df.iloc[:train_end, :]
    val_df = df.iloc[train_end:val_end, :]
    test_df = df.iloc[val_end:, :]
    
    # Create dictionaries as datasets
    documents = {}
    documents['train'] = train_df['documents'].values
    documents['val'] = val_df['documents'].values
    documents['test'] = test_df['documents'].values

    labels = {}
    labels['train'] = train_df['labels'].values
    labels['val'] = val_df['labels'].values
    labels['test'] = test_df['labels'].values

    datasets = {}
    datasets['train'] = create_tf_dataset(
        documents['train'],
        labels['train'],
        seed=SEED)

    datasets['val'] = create_tf_dataset(
        documents['val'], 
        labels['val'],
        seed=SEED)

    datasets['test'] = create_tf_dataset(
        documents['test'], 
        labels['test'],
        seed=SEED)

    return datasets, documents, labels