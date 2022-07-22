import syllables as syl
from nltk import RegexpTokenizer
import numpy as np
from scipy import sparse
import spacy
nlp = spacy.load('de_core_news_sm')

tokenizer = RegexpTokenizer(r'\w+')
umlaut_mapping = {'ü': 'ue', 'ä': 'ae', 'ö': 'oe'}

statistical_features = ['perc_syl_3', 'avg_sent_len', 'perc_word_6', 'perc_syl_1',
                  'avg_syl_count', 'tree_depth', 'fre_amstad', 'SMOG', 'wstf_1',
                  'wstf_2', 'wstf_3', 'wstf_4']


def preprocess(word):
    word = word.lower()
    for orig, replace in umlaut_mapping.items():
        word = word.replace(orig, replace)
    return word


def analyse_general_statistics(words):
    num_words = len(words)
    syls = np.array([syl.estimate(preprocess(word)) for word in words])
    perc_syl_3 = len(syls[syls >= 3]) / num_words
    perc_syl_1 = len(syls[syls == 1]) / num_words
    avg_sent_len = num_words
    word_len = np.array([len(w) for w in words])
    perc_word_6 = len(word_len[word_len > 6]) / num_words
    avg_syl_count = syls.mean()

    return perc_syl_3, avg_sent_len, perc_word_6, perc_syl_1, avg_syl_count


def fkgl(num_words, num_syls, num_sents=1):
    return (180 - (num_words / num_sents)) - 58.5 * (num_syls / num_words)


def wstf1(words):
    ms, sl, iw, es, _ = analyse_general_statistics(words)
    return 0.1935*ms + 0.1672*sl + 0.1297*iw - 0.0327*es - 0.875


def wstf2(words):
    ms, sl, iw, _, _ = analyse_general_statistics(words)
    return 0.2007*ms + 0.1682*sl + 0.1373*iw - 2.779


def wstf3(words):
    ms, sl, _, _, _ = analyse_general_statistics(words)
    return 0.2963*ms + 0.1905*sl - 1.1144


def wstf4(words):
    ms, sl, _, _, _ = analyse_general_statistics(words)
    return 0.2744*ms + 0.2656*sl - 1.693


def parse_tree_max_depth(node):
    if node.n_lefts + node.n_rights > 0:
        return 1 + max(parse_tree_max_depth(child) for child in node.children)
    else:
        return 1


def fre_amstad(general_stats):
    return 180 - general_stats[1] - (58.5 * general_stats[4])


def SMOG(words):
    syls = np.array([syl.estimate(preprocess(word)) for word in words])
    count_syl_3 = len(syls[syls >= 3])
    count_syl_3 = np.sqrt(count_syl_3)
    return 1.0430 * count_syl_3 + 3.1291


def calculate_statistics(df, sentence_column='Sentence', csr_format=True):
    # https://klartext.uni-hohenheim.de/hix
    stats = {'tree_depth': [], 'general': [], 'wstf': [], 'fre_amstad': [], 'SMOG': []}
    for i, row in df.iterrows():
        # calculate tree depth
        doc = nlp(row[sentence_column])
        depth = parse_tree_max_depth(list(doc.sents)[0].root)
        stats['tree_depth'].append(depth)
        # calculate sentence statistics
        words = tokenizer.tokenize(row[sentence_column])
        general_stats = analyse_general_statistics(words)
        stats['general'].append(general_stats)
        stats['fre_amstad'].append(fre_amstad(general_stats))
        stats['SMOG'].append(SMOG(words))
        # Wiener Sachtext Formel
        stats['wstf'].append([wstf1(words), wstf2(words), wstf3(words), wstf4(words)])

    if not csr_format:
        return stats

    stats_csr = {}
    for key, val in stats.items():
        stats_csr[key] = sparse.csr_matrix(val).reshape((len(df), -1))

    return stats_csr