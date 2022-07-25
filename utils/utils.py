import pickle
import torch
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, f1_score
from zipfile import ZipFile
from text_statistics import calculate_statistics
from transformers import EvalPrediction
from settings import sentence_column, mos_column


class TextComplexityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, df, numerical_feats, labels_col):
        self.df = df
        self.encodings = encodings
        self.numerical_feats = np.array(df[numerical_feats])
        self.labels = df[labels_col].values if labels_col is not None else torch.zeros(len(df))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __get_labels__(self):
        return self.labels

    def __len__(self):
        return len(self.df)


def get_df_with_statistics(df, from_file=True):
    if from_file:
        df = pd.read_csv(df)
    data_statistics = calculate_statistics(df, csr_format=False)
    data_statistics = pd.DataFrame(data_statistics)
    data_statistics[['perc_syl_3', 'avg_sent_len', 'perc_word_6', 'perc_syl_1', 'avg_syl_count']] = pd.DataFrame(data_statistics['general'].tolist(), index=data_statistics.index)
    data_statistics[['wstf_1', 'wstf_2', 'wstf_3', 'wstf_4']] = pd.DataFrame(data_statistics['wstf'].tolist(), index=data_statistics.index)
    del data_statistics['general']
    del data_statistics['wstf']
    df = pd.concat([df, pd.DataFrame(data_statistics)], axis=1)
    return df


def load_and_calc_statistics(df_path, balance_df=None, has_labels=True):
    df = pd.read_csv(df_path)
    df_balanced = balance(df, balance_df, mos_column) if balance_df else df
    X = df_balanced[sentence_column]
    y = df_balanced[mos_column].tolist() if has_labels else None

    statistics = calculate_statistics(df_balanced, sentence_column)
    statistics = sparse.hstack(list(statistics.values()))
    return X, y, statistics, df


def load_data(data_path, has_label=False, only_statistics=False, neural_embedding=True, neural_embedding_path=None, tfidf_vectorizer=None):
    if neural_embedding:
        assert neural_embedding_path is not None, "Please specify the path to the neural embedding or change neural_embedding to False"
    else:
        assert tfidf_vectorizer is not None, "Please specify the path to the tfidf vectorizer or change neural_embedding to True"

    X, y, stat_scores, X_df = load_and_calc_statistics(data_path, has_labels=has_label)
    if only_statistics:
        X_vec = stat_scores
    else:
        if neural_embedding:
            X_vec = np.load(neural_embedding_path)[:, 0, :]
        else:
            with open(tfidf_vectorizer, 'rb') as tfidf_in:
                vectorizer = pickle.load(tfidf_in)
            X_vec = vectorizer.transform(X)
        X_vec = sparse.hstack([X_vec, stat_scores])
    return X_vec, y, X_df


def scatter_preds(y_true, y_predicted, title=None, interpolate=False):
    plt.scatter(y_true, y_predicted)
    if interpolate:
        f = interp1d(y_true, y_predicted)
        xnew = np.linspace(1, max(y_true), num=41, endpoint=True)
        plt.plot(xnew, f(xnew), color='r')
    plt.xlabel('Correct')
    plt.ylabel('Predicted')
    plt.ylim(0.5, 7.0)
    plt.title(title)
    plt.show()


def metric_eval(y_true, y_predicted, classification):
    if classification:
        return f1_score(y_true, y_predicted, average='macro')
    else:
        return np.sqrt(mean_squared_error(y_true, y_predicted))


def write_answer_file(ids, preds):
    answer = pd.DataFrame()
    answer['ID'] = ids
    answer['MOS'] = preds

    answer.to_csv('answer.csv', index=False)
    with ZipFile('../answer.zip', 'w') as zipf:
        zipf.write('answer.csv', arcname='answer.csv')


def balance(df, num_samples, mos_column):
    df_balanced = df[df[mos_column] < 2]
    if len(df_balanced) > num_samples:
        df_balanced = df_balanced.sample(num_samples)
    for i in range(2, 7):
        filtered = df[(df[mos_column] >= i) & (df[mos_column] < i+1)]
        df_balanced = pd.concat([df_balanced, filtered]) if len(filtered) <=num_samples else pd.concat([df_balanced, filtered.sample(num_samples)])
    return df_balanced


def rmse_mapped_direct(y_true, y_pred):
    coef = np.polyfit(y_pred, y_true, 3)
    mapping = np.poly1d(coef)

    y_mapped = mapping(y_pred)
    return metric_eval(y_true, y_mapped, classification=False)

def rmse_mapped(preds: EvalPrediction):
    y_pred = preds.predictions[0].flatten()
    y_true = preds.label_ids
    return rmse_mapped_direct(y_true, y_pred)


def eval_neural_model(y_pred, labels, visualisation=True, vis_title=''):
    print('\nRMSE:', metric_eval(labels, y_pred[0], classification=False))
    print('RMSE mapped:', rmse_mapped(EvalPrediction(predictions=y_pred, label_ids=labels)))
    if visualisation:
        scatter_preds(labels, y_pred[0], vis_title)
