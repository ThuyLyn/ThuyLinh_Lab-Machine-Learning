from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

SEED = 42
REMOVE = ("headers", "footers", "quotes")  

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def ensure_dirs():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)


def load_data(remove=REMOVE):
    train = fetch_20newsgroups(subset="train", remove=remove)
    test = fetch_20newsgroups(subset="test", remove=remove)
    return train, test


def make_vectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8,
                    sublinear_tf=True, max_features=50000):
    return TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=min_df,
                           max_df=max_df, sublinear_tf=sublinear_tf,
                           max_features=max_features)


def split_validation(train, test_size=0.2):
    return train_test_split(train.data, train.target, test_size=test_size,
                            stratify=train.target, random_state=SEED)


def eda(train, save_path=None):
    names = train.target_names
    df = pd.DataFrame({"text": train.data, "y": train.target})
    df["label"] = df.y.map(lambda i: names[i])
    df["n_words"] = df.text.str.split().str.len()

    print(f"train: {len(df)} | số lớp: {len(names)}")
    print(f"Văn bản rỗng: {(df.n_words == 0).sum()} | dưới 20 từ: {(df.n_words < 20).sum()}")
    print(df.n_words.describe().round(1).to_string())

    cv = CountVectorizer(stop_words="english", min_df=5)
    counts = np.asarray(cv.fit_transform(train.data).sum(axis=0)).ravel()
    top20 = pd.Series(counts, index=cv.get_feature_names_out()).nlargest(20)
    print("20 từ phổ biến nhất:", ", ".join(f"{w}({c})" for w, c in top20.items()))

    if save_path is not None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        df.label.value_counts().sort_values().plot.barh(ax=ax[0], title="Số văn bản mỗi lớp")
        df.n_words.clip(upper=1000).plot.hist(
            bins=50, ax=ax[1], title="Độ dài văn bản (số từ, cắt ở 1000)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
    return df, top20
