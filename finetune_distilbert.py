from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.utils import rmse_mapped, write_answer_file, eval_neural_model
import datasets
from utils.settings import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ms', '--save_model_path', type=str, default=None,
                    help='Specify the path where trained model will be saved')
parser.add_argument('-ts', '--save_trainer_path', type=str, default='models/trainer',
                    help='Specify the path where the trainer output will be saved')
parser.add_argument('-w', '--write_predictions_to_file', action='store_true',
                    help='Save model predictions as answer file')
parser.add_argument('-f', '--predict_final', action='store_true',
                    help='Run predictions on final test data instead of validation data')
parser.add_argument('-v', '--no_visualization', action='store_true',
                    help='If not set, predictions will be printed in scatter plot')
parser.add_argument('-e', '--epochs', type=int, default=2,
                    help='Define the number of fine-tuning epochs')
args = parser.parse_args()


#save_model_path = 'models/distilbert_final'
print('Load tokenizer')
tokenizer = AutoTokenizer.from_pretrained(base_model_string)
def tokenize_data(example):
    return tokenizer(example[sentence_column])

print('Load and tokenize training data')
data = datasets.load_dataset('csv', data_files=training_data_path)
data = data['train'].rename_columns({label_column: 'label'})

data = data.map(tokenize_data, batched=True)
data = data.train_test_split(test_size=0.05)

print('Load base model. This will show a warning that some weights were not loaded but this is fine as we are performing fine-tuning!')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_string, num_labels=1,
                                                                output_hidden_states=True, return_dict=True)

training_args = TrainingArguments(args.save_trainer_path, num_train_epochs=args.epochs)
trainer = Trainer(model=base_model, args=training_args, tokenizer=tokenizer, train_dataset=data['train'],
                  eval_dataset=data['test'], compute_metrics=rmse_mapped)
trainer.train()
trainer.evaluate()

print('Evaluating model performance')

y_pred_train = trainer.predict(test_dataset=data['train'])[0]
y_pred_test = trainer.predict(test_dataset=data['test'])[0]

eval_neural_model(y_pred_train, np.array(data['train']['label']), not args.no_visualisation, 'Training')
eval_neural_model(y_pred_test, np.array(data['test']['label']), not args.no_visualisation, 'Test')


if args.save_model_path is not None:
    print('Saving model')
    trainer.save_model(args.save_model_path)


if args.write_predictions_to_file:
    print('Load validation data')
    if args.predict_final:
        validation_data_path = final_test_path
        validation_feature_path = final_test_feature_path
    data_val = datasets.load_dataset('csv', data_files=validation_data_path)
    data_val = data_val.map(tokenize_data, batched=True)
    preds = trainer.predict(test_dataset=data_val['train'])
    write_answer_file(data_val['train']['ID'], preds[0][0])

