training_data_path = '../data/training_set_balanced.csv'
training_feature_path = '../data/training_set_balanced_distilbert_final.npy'
validation_data_path = '../data/validation_set.csv'
validation_feature_path = '../data/validation_set_distilbert_final.npy'
final_test_path = '../data/part2_public.csv'
final_test_feature_path = '../data/final_set_distilbert_final.npy'
tfidf_vectorizer_path = '../models/wiki_tfidf.pkl'

sentence_column = 'Sentence'
mos_column = 'MOS'

base_model_string = "distilbert-base-german-cased"
seq_max_len = 99 #this value is given by the tokenizer training, leave it untouched for our models