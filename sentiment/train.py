"""
Train models
"""

# Imports ---------------------------------------------------------------------

import datetime
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GRU 
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sentiment import data
from sentiment.constants import LOG_DIR
from sentiment.constants import MODEL_DIR
from sentiment.constants import VECTORIZER_DIR
from sentiment.constants import VOCAB_SIZE
from sentiment.constants import SEQUENCE_LENGTH
from sentiment.constants import EMBEDDING_DIM

# Vectorizarion ---------------------------------------------------------------

def get_vectorizer(train_documents):
    vectorizer = TextVectorization(
        standardize='lower_and_strip_punctuation',
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)
    vectorizer.adapt(train_documents)
    return vectorizer

# Model -----------------------------------------------------------------------

def get_embedding_model(vectorizer):

    text_input = Input(shape=(1,), dtype='string', name='text_input')
    text_vectorized = vectorizer(text_input)
    text_embedding = Embedding(
        input_dim = VOCAB_SIZE, 
        output_dim = EMBEDDING_DIM)(text_vectorized)
    text_pooling = GlobalAveragePooling1D(name='text_pooling')(text_embedding)
    dropout_1 = Dropout(rate=0.2, name='dropout_1')(text_pooling)
    dense_1 = Dense(128, activation='relu', name='dense_1')(dropout_1)
    dropout_2 = Dropout(rate=0.2, name='dropout_2')(dense_1)
    dense_2 = Dense(128, activation='relu', name='dense_2')(dropout_2)
    dropout_3 = Dropout(rate=0.2, name='dropout_3')(dense_2)
    output = Dense(1, name='output')(dropout_3)
    model = Model(inputs=text_input, outputs=output)

    return model


def get_gru_sequence_model(vectorizer):

    text_input = Input(shape=(1,), dtype='string', name='text_input')
    text_vectorized = vectorizer(text_input)
    text_embedding = Embedding(
        input_dim = VOCAB_SIZE, 
        output_dim = EMBEDDING_DIM)(text_vectorized)
    gru_sequences = GRU(
        units=128, 
        return_sequences=True, 
        dropout=0.2,
        recurrent_dropout=0.2,
        name='gru_sequences')(text_embedding)
    gru_final = GRU(
        units=128, 
        dropout=0.2,
        recurrent_dropout=0.2,
        name='gru_final')(gru_sequences)
    output = Dense(1, name='output')(gru_final)
    model = Model(inputs=text_input, outputs=output)

    return model

# Fit -------------------------------------------------------------------------

def fit_model(
    train_dataset,
    val_dataset,
    vectorizer,
    model_name,
    epochs=20,
    patience=6):

    # Define paths to save model data
    model_path = os.path.join(
        MODEL_DIR, '{0}'.format(model_name))
    
    log_path = os.path.join(
        LOG_DIR, model_name, '{0}-{1}'.format(
            model_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    
    # Get the model
    model = get_gru_sequence_model(vectorizer)

    model.compile(
        loss=BinaryCrossentropy(from_logits=True), 
        optimizer=Adam(learning_rate=0.00005),
        metrics=['accuracy'])

    model.summary()

    # Create callback for early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
        restore_best_weights=True,
        verbose=1)

    # Create callback for saving the model
    model_checkpoint = ModelCheckpoint(
        model_path, 
        monitor='val_loss', 
        mode='min', 
        verbose=1,
        save_best_only=True, 
        save_freq='epoch')

    # Create callback for logging tensorboard data
    tensorboard = TensorBoard(log_dir=log_path)

    # Define callbacks
    callbacks = [early_stopping, model_checkpoint, tensorboard]

    # Fit model
    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=epochs, 
        callbacks=callbacks)

    # Save vectorizer for the model
    save_vectorizer(vectorizer, model_name)

    return model, history

# Train -----------------------------------------------------------------------

def train_model(model_name):

    # Load the training data
    datasets, documents, labels = data.load_dataset()

    # Get vectorization layer
    vectorizer = get_vectorizer(documents['train'])

    # Count the classes
    n_classes = len(set(labels['train']))

    model, history = fit_model(
        train_dataset=datasets['train'],
        val_dataset=datasets['val'],
        vectorizer=vectorizer,
        model_name=model_name)

    return model, history, vectorizer


# Save model ------------------------------------------------------------------

def save_vectorizer(vectorizer, model_name):

    """
    The vecortizer must be saved separately from the model in order to properly 
    reconstruct the full model when loading it from disk. This is necessary 
    because of a bug in how TextVectorization objects are serialised. See ...
    https://stackoverflow.com/questions/70255845/
    """

    vectorizer_path = os.path.join(
        VECTORIZER_DIR, '{}_vectorizer.pkl'.format(model_name))

    vectorizer_config = {
        'config': vectorizer.get_config(),
        'weights': vectorizer.get_weights()}

    pickle.dump(vectorizer_config, open(vectorizer_path, 'wb'))


# Load model ------------------------------------------------------------------    

def load_model_with_vectorizer(model_name, train_documents):

    """
    In order to load the model from disk, it must be reconstructed from the 
    model data saved during training and the vectorizer, which is saved 
    separately. This is necessary because of a bug in how TextVectorization
    objects are serialised. See ...
    https://stackoverflow.com/questions/70255845/
    """

    # Load vectorizer from disk
    vectorizer_path = os.path.join(
        VECTORIZER_DIR, '{}_vectorizer.pkl'.format(model_name))
    
    vec_disk = pickle.load(open(vectorizer_path, 'rb'))

    # Initialise a new vectrorizer using the data from disk
    vectorizer = TextVectorization(
        max_tokens=vec_disk['config']['max_tokens'],
        output_mode='int',
        output_sequence_length=vec_disk['config']['output_sequence_length'])

    vectorizer.adapt(train_documents)
    vectorizer.set_weights(vec_disk['weights'])

    # Load the model from disk
    model_path = os.path.join(
        MODEL_DIR, '{0}'.format(model_name))

    model_disk = load_model(model_path)

    model = Sequential()
    model.add(vectorizer)
    model.add(model_disk.layers[2])
    model.add(model_disk.layers[3])
    model.add(model_disk.layers[4])
    model.add(model_disk.layers[5])

    return model, vectorizer
