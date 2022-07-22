from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from utils import load_data, rmse_mapped_direct, scatter_preds, write_answer_file, metric_eval
import joblib
import argparse
from settings import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--no_neural_embedding', action='store_false', dest='neural_embedding',
                    help='Set to True if DistilBERT embedding should be used and False for TFIDF embedding')
parser.add_argument('-s', '--only_statistics', action='store_true',
                    help='Use only test statistics as features')
parser.add_argument('-w', '--write_predictions_to_file', action='store_true',
                    help='Save model predictions as answer file')
parser.add_argument('-f', '--predict_final', action='store_true',
                    help='Run predictions on final test data instead of validation data')
parser.add_argument('-m', '--save_model_path', type=str, default=None,
                    help='Specify the path where trained model will be saved')
parser.add_argument('-l', '--load_model_path', type=str, default='models/svr_distil_statistics.pkl',
                    help='Path to a pretrained model that should be used')
parser.add_argument('-t', '--training_mode', action='store_true',
                    help='Specifies that a new model should be trained')
parser.add_argument('-v', '--no_visualization', action='store_true',
                    help='If not set, predictions will be printed in scatter plot')
args = parser.parse_args()

# 1. Load data from path and calculate statistics

print('Load training data')
X_train_vec, y_train, df_train = load_data(training_data_path, neural_embedding_path=training_feature_path,
                                           neural_embedding=args.neural_embedding, only_statistics=args.only_statistics,
                                           tfidf_vectorizer=tfidf_vectorizer_path, has_label=True)

if args.training_mode:
    X_train_vec, X_test_vec, y_train, y_test = train_test_split(X_train_vec, y_train, test_size=0.05, shuffle=True)

    print('Fitting SVR')
    clf = SVR()
    clf.fit(X_train_vec, y_train)

    if args.save_model_path is not None:
        print('Saving model to file')
        joblib.dump(clf, args.save_model_path)

    y_pred_test = clf.predict(X_test_vec)
    print('Test RMSE:', metric_eval(y_test, y_pred_test, False))
    print('Test RMSE mapped:', rmse_mapped_direct(y_test, y_pred_test))
    if not args.no_visualization:
        scatter_preds(y_test, y_pred_test, 'Test data')
else:
    print('Load model from file')
    clf = joblib.load(args.load_model_path)

print('Evaluating model')
y_pred_train = clf.predict(X_train_vec)
print('Train RMSE:', metric_eval(y_train, y_pred_train, False))
print('Train RMSE mapped:', rmse_mapped_direct(y_train, y_pred_train))
if not args.no_visualization:
    scatter_preds(y_train, y_pred_train, 'Trainig data')

if args.write_predictions_to_file:
    print('Load validation data')
    if args.predict_final:
        validation_data_path = final_test_path
        validation_feature_path = final_test_feature_path
    X_val_vec, _, df_val = load_data(validation_data_path, neural_embedding_path=validation_feature_path,
                                     neural_embedding=args.neural_embedding, only_statistics=args.only_statistics,
                                     tfidf_vectorizer=tfidf_vectorizer_path)
    y_pred_val = clf.predict(X_val_vec)
    print('Writing predictions to file')
    write_answer_file(df_val.ID, y_pred_val)
