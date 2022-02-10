"""
High-level interface to reproduce the model and the explanatory data.
"""

# Imports ---------------------------------------------------------------------

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sentiment import data
from sentiment import train
from sentiment import shap

# Train a model ---------------------------------------------------------------

def train_model(model_name):
    model, history, vectorizer = train.train_model(model_name)
    return model, history, vectorizer

# Evaluate a model ------------------------------------------------------------

def evaluate_model(model_name, evaluation_set='test'):
    
    datasets, documents, labels = data.load_dataset()
    model, vectorizer = train.load_model_with_vectorizer(
        model_name, documents['train'])
    
    y_logits = model.predict(documents[evaluation_set])
    y_pred = (y_logits > 0).astype(int)
    y_true = labels[evaluation_set]

    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision'] = precision_score(y_true, y_pred)
    results['recall'] = recall_score(y_true, y_pred)
    results['f1_score'] = f1_score(y_true, y_pred)

    return results

# Get SHAP values and other explanatory data ----------------------------------

def explain_model(
    model_name,
    evaluation_set='test', 
    evaluation_start=0, 
    evaluation_end=10,
    background_start=0, 
    background_end=100):
    
    datasets, documents, labels = data.load_dataset()
    model, vectorizer = train.load_model_with_vectorizer(
        model_name, documents['train'])
    
    train_docs = documents['train'][background_start:background_end]
    explain_docs = documents[evaluation_set][evaluation_start:evaluation_end]
    explain_labs = labels[evaluation_set][evaluation_start:evaluation_end]

    explain_predictions, explain_data, shap_values = shap.explain_predictions(
        model, 
        vectorizer, 
        train_docs,
        explain_docs)

    reviews = shap.get_explanatory_data(
        vectorizer, 
        explain_docs, 
        explain_labs,
        explain_predictions, 
        shap_values)

    return reviews