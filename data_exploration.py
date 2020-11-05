import numpy as np
import pandas as pd
from tabulate import tabulate
import math

import seaborn as sns
import nltk
from nltk import ngrams
import collections

def validate_null_value(news_df):

    print(tabulate(news_df.head(5), headers='keys', tablefmt='psql'))
    print(news_df.shape)

    # check for a column with any null value
    print("List the count of null values for each column")
    sum_null_value = news_df.isnull().sum()
    print(sum_null_value)

    # validate if there is any null value
    bool = news_df.isnull().values.any()
    print("Is there any null value? " + str(bool))

    # check for data type
    print("Display data type of all columns")
    print(news_df.dtypes)
    print(news_df.info)


'''
def claim_version_combine(df_claim_v1, df_claim_v2):
    header = ['Id', 'fact_check_url', 'news_url', 'title']
    df_claim_v1.columns = header
    print(tabulate(df_claim_v1.head(5), headers='keys', tablefmt='psql'))
    print(df_claim_v1.shape)
    df_claim_v2.columns = header
    print(tabulate(df_claim_v2.head(5), headers='keys', tablefmt='psql'))
    print(df_claim_v2.shape)
    df_claim = pd.concat([df_claim_v1, df_claim_v2])
    print(tabulate(df_claim.head(5), headers='keys', tablefmt='psql'))
    print(df_claim.shape)
    return df_claim


def news_version_combine(df_news_v1, df_news_v2):
    df_news_v1.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
    df_news_v2.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
    print(tabulate(df_news_v1.head(5), headers='keys', tablefmt='psql'))
    print(df_news_v1.shape)
    print(tabulate(df_news_v2.head(5), headers='keys', tablefmt='psql'))
    print(df_news_v2.shape)
    df_news = pd.concat([df_news_v1, df_news_v2])
    print(tabulate(df_news.head(5), headers='keys', tablefmt='psql'))
    print(df_news.shape)
    return df_news


def tweets_version_combine(df_tweets_v1, df_tweets_v2):
    print(tabulate(df_tweets_v1.head(5), headers='keys', tablefmt='psql'))
    print(df_tweets_v1.shape)
    print(tabulate(df_tweets_v2.head(5), headers='keys', tablefmt='psql'))
    print(df_tweets_v2.shape)
    df_tweets = pd.concat([df_tweets_v1, df_tweets_v2])
    print(tabulate(df_tweets.head(5), headers='keys', tablefmt='psql'))
    print(df_tweets.shape)
    return df_tweets


def retweets_version_combine(df_retweets_v1, df_retweets_v2):
    print(tabulate(df_retweets_v1.head(5), headers='keys', tablefmt='psql'))
    print(df_retweets_v1.shape)
    print(tabulate(df_retweets_v2.head(5), headers='keys', tablefmt='psql'))
    print(df_retweets_v2.shape)
    df_retweets = pd.concat([df_retweets_v1, df_retweets_v2])
    print(tabulate(df_retweets.head(5), headers='keys', tablefmt='psql'))
    print(df_retweets.shape)
    return df_retweets


def merge_main_content_tweets(main_content, tweets):
    tweets['tweet_count'] = tweets.groupby(['index']).transform('count')
    tweets = tweets.drop_duplicates(subset=['index']).reset_index()
    tweets.drop("level_0", axis=1, inplace=True)
    merged_content = main_content.merge(tweets, left_on='Id', right_on='index', how='left')
    merged_content.drop("index", axis=1, inplace=True)
    print(tabulate(merged_content.head(5), headers='keys', tablefmt='psql'))
    print(merged_content.shape)
    return merged_content

def merge_main_content_retweets(main_content, retweets):
     retweets['retweet_count'] = retweets.groupby(['tweet_id'])['reply_id'].transform('count')
     retweets = retweets.drop_duplicates(subset=['news_id']).reset_index()
     retweets.drop("index", axis=1, inplace=True)
     retweets.drop("tweet_id", axis=1, inplace=True)
     merged_content = main_content.merge(retweets, left_on='Id', right_on='news_id', how='left')
     merged_content.drop("news_id", axis=1, inplace=True)
     print(tabulate(merged_content.head(5), headers='keys', tablefmt='psql'))
     print(merged_content.shape)
     return merged_content


def claim_preprocess(claim_fake, claim_real):
    claim_fake['isfake'] = 1
    claim_real['isfake'] = 0
    claim = pd.concat([claim_fake, claim_real])
    print(tabulate(claim.head(5), headers='keys', tablefmt='psql'))
    print(claim.shape)

    # check for data type
    print("Display data type of all claim columns")
    print(claim.dtypes)

    claim['tweet_id'] = claim['tweet_id'].apply(lambda x: np.random.uniform(0.1, 0.5) if math.isnan(x) else x)
    claim['reply_id'] = claim['reply_id'].apply(lambda x: np.random.uniform(0.1, 0.5) if math.isnan(x) else x)
    claim['tweet_count'] = claim['tweet_count'].apply(lambda x: 0 if math.isnan(x) else x)
    claim['retweet_count'] = claim['retweet_count'].apply(lambda x: 0 if math.isnan(x) else x)
    print(tabulate(claim.head(5), headers='keys', tablefmt='psql'))
    return claim


def news_preprocess(news_fake, news_real):
    news_fake['isfake'] = 1
    news_real['isfake'] = 0
    news = pd.concat([news_fake, news_real])
    print(tabulate(news.head(5), headers='keys', tablefmt='psql'))
    print(news.shape)

    # check for data type
    print("Display data type of all news columns")
    print(news.dtypes)

    news['tweet_id'] = news['tweet_id'].apply(lambda x: np.random.uniform(0.1, 0.5) if math.isnan(x) else x)
    news['reply_id'] = news['reply_id'].apply(lambda x: np.random.uniform(0.1, 0.5) if math.isnan(x) else x)
    news['tweet_count'] = news['tweet_count'].apply(lambda x: 0 if math.isnan(x) else x)
    news['retweet_count'] = news['retweet_count'].apply(lambda x: 0 if math.isnan(x) else x)
    print(tabulate(news.head(5), headers='keys', tablefmt='psql'))
    return news
'''


def word_frequency(text: str):
    """
    Get frequency of individual words in a text corpus

    :param text: text corpus
    :return: Counter object of tuples of words and their frequencies
    """
    # get individual words
    tokenized = text.split()

    # count the frequency
    word_counter = collections.Counter(tokenized)
    #frequencies = list(collections.Counter(tokenized).items())

    return word_counter


def ngram_frequency(text: str, n: int = 2):
    """
    Get ngrams frequency in a text corpus

    :param text: the text of interest
    :param n: n for ngram. Default = 2
    :return: Counter object of tuples of ngrams and their frequencies
    """
    # get individual words
    tokenized = text.split()

    # get the list of all the ngrams
    ngram_list = ngrams(tokenized, n)

    # get the frequency of each ngram in the corpus
    ngram_counter = collections.Counter(ngram_list)

    return ngram_counter
