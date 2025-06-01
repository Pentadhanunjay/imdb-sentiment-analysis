IMDb Sentiment Analysis Project 🎬

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



##  Insights

- Combining interpretable models (LR) with powerful learners (XGBoost, SVM) improves robustness.
- Cleaning noisy data and removing dominant stopwords has a measurable impact on performance.
- Bayesian optimization outperforms grid search for XGBoost hyperparameter tuning.



##  Results

Achieved a **mean F1-score of ~0.88** using stratified cross-validation on filtered, optimized data.





