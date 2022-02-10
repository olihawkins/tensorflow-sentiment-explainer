"""
Get explanatory data.
"""

# Imports ---------------------------------------------------------------------

import numpy as np

from shap import KernelExplainer
from tensorflow.keras.models import Sequential

from sentiment.constants import SEQUENCE_LENGTH

# Get a function that predicts using a model without a vectorizer -------------

def get_submodel_func(model):
    submodel = Sequential()
    for layer in model.layers[1:]:
        submodel.add(layer)
    def submodel_func(input):
        output = submodel.predict(input)
        return output
    return submodel_func

# Create an explainer using a given model and its vectorizer ------------------

def get_explainer(model, vectorizer, train_documents):
    submodel_func = get_submodel_func(model)
    background_data = vectorizer(train_documents).numpy()
    explainer = KernelExplainer(submodel_func, background_data)
    return explainer

# Make predictions and explain them -------------------------------------------

def explain_predictions(
    model, 
    vectorizer, 
    train_documents,
    explain_documents,
    nsamples=1000):

    explain_predictions = model.predict(explain_documents)
    explain_predictions = 1 / (1 + np.exp(-explain_predictions))
    explainer = get_explainer(model, vectorizer, train_documents)
    explain_data = vectorizer(explain_documents).numpy()
    shap_values = explainer.shap_values(explain_data, nsamples=nsamples)
    return explain_predictions, explain_data, shap_values[0]

# Combine data for representation ---------------------------------------------

def get_explanatory_data(
    vectorizer, 
    explain_documents, 
    explain_labels,
    explain_predictions, 
    shap_values):

    # Get the vocabulary from the vectorizer
    vocabulary = vectorizer.get_vocabulary()
    explain_data = explain_data = vectorizer(explain_documents).numpy()

    # Compile a list of data dictionaries for each review
    reviews = []

    for i in range(len(explain_documents)):
        
        review = {}
        review['document'] = explain_documents[i]
        review['label'] =  int(explain_labels[i])
        review['prediction'] = float(explain_predictions[i][0])
        review['tokens'] = []

        for j in range(SEQUENCE_LENGTH):
            token = {}
            token['index'] = j
            token['id'] = int(explain_data[i][j])
            token['token'] = vocabulary[explain_data[i][j]]
            token['shap_value'] = float(shap_values[i][j])
            review['tokens'].append(token)

        reviews.append(review)

    return reviews
