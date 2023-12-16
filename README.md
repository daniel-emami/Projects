# constructive-classification
Master thesis on classifying constructive articles by Daniel Riis and Esben Rasmussen

# Code for models
## TF-IDF models
SpaCy_TF-IDF.ipynb contains models using TF-IDF and code for Lime explanations.

## Metadata models
trad_ML_META.ipynb contains models using metadata and feature importance plots.

## Bert model
The trained BERT model is not included in this repository as the file size is too large.
Model can be viewed BERT_final.ipynb.
Shap explanations in SHAP_explanations_BERT.ipynb

## Lime explanations
lime_explanations has a folder structure with explanations for the entire test set. The folders are named based on the index number of each article in the text set.

## Data

Articles used for classification can be found in the Data folder. There is the original data splits in train, test, val.
Dataframe for mislabelled articles can be viewed in Data/mislabelled_articles.csv

## Constructive word list
constuctive_words folder contains a list of perceived constructive words used for the metadata model.
