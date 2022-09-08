import joblib
import shap
from utils.settings import *
from utils.utils import load_data
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--combined', action='store_true',
                    help='Path to model which should be explained')
parser.add_argument('-s', '--sample', action='store_true',
                    help='Sample data to boost performance')
args = parser.parse_args()
print(args)

neural_embedding = True
if args.combined:
    model_path = combined_model_path
    only_statistics = False
    embedding_dim = 768
    max_display = 10
else:
    model_path = statistics_model_path
    only_statistics = True
    embedding_dim = 0
    max_display = 12

print('Load model from file')
clf = joblib.load(model_path)

print('Load training data')
X_train_vec, _, _ = load_data(training_data_path, neural_embedding_path=training_feature_path,
                                           neural_embedding=neural_embedding, only_statistics=only_statistics,
                                           tfidf_vectorizer=tfidf_vectorizer_path, has_label=True)
if args.sample:
    idx = np.random.randint(X_train_vec.shape[0], size=10)
    X_train_vec = X_train_vec.tocsr()[idx]

print('Building explainer and calculating SHAP values')
explainer = shap.KernelExplainer(clf.predict, X_train_vec)
shap_values = explainer.shap_values(X_train_vec.tocsr())

features_names = [f'distil_%i'%i for i in range(embedding_dim)] + \
                 ['mtd', 'ps3', 'asl', 'pw6', 'ps1', 'asc', 'wstf1', 'wstf2', 'wstf3', 'wstf4', 'fre_amstad', 'SMOG']
shap.summary_plot(shap_values, X_train_vec, feature_names=features_names, plot_type='bar', max_display=max_display)
