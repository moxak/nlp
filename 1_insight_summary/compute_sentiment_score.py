
import os, sys, glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from transformers import pipeline, Pipeline


def _compute_sentiment_score(probabilities):
    """感情得点を計算する
    感情得点：
        - 肯定的なほど1に近くなり、否定的なほど-1に近くなる。また、中立的なほど0に近くなるような指標。
    Args:
        probabilities (numpy.ndarray): 肯定的、中立的、否定的な確率の配列
            [negative_prob, positive_prob, neutral_prob]
    Returns:
        sentiment_score (float): 感情得点
    
    Examples:
        >>> import numpy as np
        >>> probabilities = np.array([0.1, 0.7, 0.2])
        >>> calculate_sentiment_score(probabilities)
        0.15042637522293034
    """

    # 感情得点の計算
    # negative_prob, positive_prob, neutral_prob = probabilities
    negative_probs = probabilities[:, 0]
    positive_probs = probabilities[:, 1]
    neutral_probs = probabilities[:, 2]
    sentiment_score = (positive_probs * (1 - neutral_probs)) - (negative_probs * (1 - neutral_probs))
    sentiment_score = sentiment_score / 2 + 0.5 # -1から1の範囲を0から1に変換

    return sentiment_score

def _log_inverted_score(scores):
    scores = np.abs(np.log(1- scores))
    return scores

def _min_max_scaling(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data


def compute_sentiment_score(df:pd.DataFrame):
    df['sentiment_score'] = _compute_sentiment_score(df[['negative_score', 'positive_score', 'neutral_score']].to_numpy())
    
    # スケールを変換する
    # Positive
    idx = df[(df['sentiment_label'] == 'positive')].index
    df.loc[idx, 'sentiment_score'] = _log_inverted_score(df.loc[idx, 'sentiment_score'])

    # Neutral
    idx = df.query('sentiment_label == "neutral"').index
    df.loc[idx, 'sentiment_score'], _ = boxcox(df.loc[idx, 'sentiment_score'])

    # Negative
    idx = df.query('sentiment_label == "negative"')['sentiment_score'].index
    df.loc[idx, 'sentiment_score'] = np.log(df.loc[idx, 'sentiment_score'])

    # 0-1の範囲に正規化
    df['sentiment_score'] = _min_max_scaling(df['sentiment_score'])
    return df


def compute_sentiment_score_by_BERT(
    df:pd.DataFrame, 
    model_path:str, 
    tokenizer_model:str='cl-tohoku/bert-large-japanese-v2') -> pd.DataFrame:
    """[answer_question]を持つデータフレームに対して、感情分析を実施し、['sentiment_score', 'sentimetn_label']を追加して返す。"""
    tokenizer = BertJapaneseTokenizer.from_pretrained(tokenizer_model)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    df['sentiment_results'] = df['answer_question'].apply(lambda x: sentiment_analyzer(x, return_all_scores=True))
    df['negative_score'] = df['sentiment_results'].apply(lambda x: x[0][0]['score'])
    df['positive_score'] = df['sentiment_results'].apply(lambda x: x[0][1]['score'])
    df['neutral_score'] = df['sentiment_results'].apply(lambda x: x[0][2]['score'])

    df['sentiment_label'] = df[['negative_score', 'positive_score', 'neutral_score', ]].idxmax(axis=1).apply(lambda x: x.split('_')[0])

    df = compute_sentiment_score(df=df)
    df.drop(['sentiment_results', 'negative_score', 'positive_score', 'neutral_score'], axis=1, inplace=True)

    return df
    
if __name__ == "__main__":
    # 単体テスト
    FT_MODEL_DIR = '../sentiment_classification/'
    compute_sentiment_score_by_BERT(model_path=os.path.join(FT_MODEL_DIR, 'results_bert-base-japanese-v3_50'))