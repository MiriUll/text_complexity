# Text Complexity Assessment of German Text
This repository contains the code for the TUM Social Computing team at the GermEval 2022 shared task.
Our SVR models were trained with Python 3.9.
## Setup
1. Clone this repository
2. Setup Python 3.9 environment
3. Install requirements with `pip install -r utils/requirements.txt`
4. Download the spacy pipeline with `python -m spacy download de_core_news_sm`
5. Download the Shared Task data either from the [competition homepage](https://codalab.lisn.upsaclay.fr/competitions/4964#participate) or the [original Github repository](https://github.com/babaknaderi/TextComplexityDE).
6. Adap the paths and column names in `utils/settings.py` to your version of the data.

## Use pretrained models
Our SVR models are uploaded in this reporsitory in the `models` folder. The fine-tuned DistilBERT model is uploaded to HuggingFace and can be found [here](https://huggingface.co/MiriUll/distilbert-german-text-complexity).
To run the respective models, use these commands from the command line
### Combination of neural embedding with text statistics
```
python support_vector_regression.py
```
### SVR with statistics only
```
python support_vector_regression.py --only_statistics
```
### Fine-tuned distilbert
```
python eval_distilbert.py
```

## Generate explanations
To analyze the relevant features in the SVR models, use the `feature_relevance_analysis.py` module.
```
python feature_relevance_analysis.py -s 
```
The `-s` flag samples to data to speed up the SHAP value calculation. If you want to evaluate the combined model, add the `-c` parameter.

## Train models
To recreate the SVR models, run the following command.
```
python support_vector_regression.py --training_mode 
```
To retrain the DistilBERT fine-tuning, use the `finetune_distilbert.py` module.