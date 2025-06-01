**IMDb Sentiment Analysis Project**🎬

This project performs **Sentiment Analysis on the IMDb Movie Review Dataset** using a combination of traditional ML and modern optimization techniques.

 Features Included

### Preprocessing
- HTML tag removal, punctuation cleaning, lowercasing
- Custom stopword removal (based on 50 most frequent words)
- Noisy review filtering (based on length and repetition)

###  Feature Extraction
- TF-IDF vectorization with unigrams and bigrams
- Top 10,000 vocabulary features

###  Modeling
- **Logistic Regression with L1** for feature selection
- **XGBoost**, tuned via **Bayesian Optimization (Optuna)**
- **Linear SVM**
- Final model is an **Ensemble (VotingClassifier)** combining all three

###  Evaluation
- **Stratified 5-Fold Cross-Validation**
- F1-score as the primary metric

###  Interpretability
- Shows **Top 5 misclassified reviews**
- Displays **top contributing words** to the misclassification

---

##  Requirements

```bash
pip install pandas scikit-learn xgboost optuna
```

Also download the IMDb dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

---

##  Files

- `imdb_sentiment_pipeline.py`: Full end-to-end pipeline script
- `IMDB Dataset.csv`: Input dataset (to be downloaded separately)

**final output**
--- Top 5 Misclassified Reviews with Contributing Words ---

Review #11: Actual: negative | Predicted: positive
I saw this movie when I was about 12 when it came out...
  loved → weight: 0.6007
  predictable → weight: 0.5225
  unintentional → weight: 0.4472
  beautiful → weight: 0.2937
  still → weight: 0.2614

Review #13: Actual: negative | Predicted: positive
The cast played Shakespeare... Shakespeare lost...
  favorite → weight: 0.7661
  appreciate → weight: 0.4221
  write → weight: 0.3521
  least → weight: 0.3404
  trying → weight: 0.2416

Review #14: Actual: positive | Predicted: negative
This a fantastic movie of three prisoners who become famous...
  fantastic → weight: 1.3839
  bad → weight: 1.1451
  soundtrack → weight: 0.2951
  become → weight: 0.2538
  actors → weight: 0.2057





