from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DistilBertModel
from utils.settings import *
import numpy as np
import argparse
from utils.text_statistics import statistical_features
from utils.utils import get_df_with_statistics, TextComplexityDataset, eval_neural_model, write_answer_file


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', default="MiriUll/distilbert-german-text-complexity", type=str,
                    help='Path to pretrained model')
parser.add_argument('-d', '--data_path', default=training_data_path, type=str,
                    help='Path to data that should be evaluated')
parser.add_argument('-e', '--embedding_path', default=training_feature_path, type=str,
                    help='Path to store the extracted embeddings')
parser.add_argument('--embedding', action='store_true',
                    help='Set if neural embedding should be calculated instead of predictions')
parser.add_argument('-v', '--no_visualization', action='store_true',
                    help='If not set, predictions will be printed in scatter plot')
parser.add_argument('-w', '--write_predictions_to_file', action='store_true',
                    help='Save model predictions as answer file')
args = parser.parse_args()

# Extract features
print('Load pretrained model and tokenizer')
if args.embedding:
    pretrained = DistilBertModel.from_pretrained(args.model_path)
else:
    pretrained = DistilBertForSequenceClassification.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_string)
feature_trainer = Trainer(model=pretrained, args=TrainingArguments('models/feature_trainer', num_train_epochs=2),
                          tokenizer=tokenizer) # for compatability reasons, we use trainer for predictions

print('Load and tokenize data')
data_df = get_df_with_statistics(args.data_path)
data_tokenized = tokenizer(list(data_df[sentence_column]), padding='max_length', truncation=True, max_length=seq_max_len)
data_complexity = TextComplexityDataset(data_tokenized, data_df, statistical_features, label_column)

print('Run predictions')
preds = feature_trainer.predict(test_dataset=data_complexity)

if args.embedding:
    np.save(args.embedding_path, preds[0][0])
else:
    eval_neural_model(preds[0], data_complexity.__get_labels__(), not args.no_visualization)
    if args.write_predictions_to_file:
        write_answer_file(data_df.ID, preds[0][0])
