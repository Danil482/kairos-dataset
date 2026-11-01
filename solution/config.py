SEED = 42

DENSE_PARAMS = {
    "objective": "multiclass", 
    "num_class": 3,
    "n_estimators": 800, 
    "learning_rate": 0.04, 
    "num_leaves": 63,
    "min_child_samples": 20, 
    "subsample": 0.8, 
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1, 
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "importance_type": "gain",
    "random_state": SEED, 
    "n_jobs": -1, 
    "verbose": -1,
}

TEXT_PARAMS = {
    "objective": "multiclass", 
    "num_class": 3,
    "n_estimators": 300, 
    "learning_rate": 0.05, 
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8, 
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": SEED, 
    "n_jobs": -1, 
    "verbose": -1,
}

WORD_TFIDF = {
    "ngram_range": (1, 2), 
    "max_features": 50000,
    "min_df": 3, 
    "max_df": 0.9,
    "sublinear_tf": True, 
    "strip_accents": "unicode",
}

CHAR_TFIDF = {
    "analyzer": "char_wb", 
    "ngram_range": (3, 5),
    "max_features": 50000, 
    "min_df": 3, 
    "max_df": 0.9,
    "sublinear_tf": True,
}

SVD_DIMS = 256

BLEND_DENSE = 0.55
BLEND_TEXT = 0.45