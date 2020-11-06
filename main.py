import numpy as np
import pandas as pd

from data_compilation import (
    build_real_news_dataframe,
    build_fake_news_dataframe,
    concatenet_dataframes
)

from data_exploration import (
    # claim_version_combine,
    # news_version_combine,
    # tweets_version_combine,
    # retweets_version_combine,
    # merge_main_content_tweets,
    # merge_main_content_retweets,
    # claim_preprocess,
    # news_preprocess,
    validate_null_value,
    ngram_frequency,
    word_frequency
)

from model_building import (
    decision_tree_model,
    make_prediction
)

def main():
    #data loading
    real_news_df = build_real_news_dataframe(include_celeb=True)
    fake_news_df = build_fake_news_dataframe(include_celeb=True)
    news_df = concatenet_dataframes(real_news_df, fake_news_df)

    # find missing values
    validate_null_value(news_df)

    # concatenate all contents of real news
    real_news_contents = np.array(real_news_df['Content'].tolist())
    real_news_joined = " ".join(real_news_contents)

    # Find ngrams
    real_news_ngram_fred = ngram_frequency(real_news_joined, 3)
    print('\n')
    print("Frequency of the 20 most common ngrams in real news:")
    print(real_news_ngram_fred.most_common(20))
    print('\n')

    # find frequencies of words
    real_news_word_fred = word_frequency(real_news_joined)
    print("Frequency of the 20 most common words in real news:")
    print(real_news_word_fred.most_common(20))

    # --------------------------------------------------------------------
    # concatenate all contents of fake news
    fake_news_contents = np.array(fake_news_df['Content'].tolist())
    fake_news_joined = " ".join(fake_news_contents)

    # Find ngrams
    fake_news_ngram_fred = ngram_frequency(fake_news_joined, 3)
    print('\n')
    print("Frequency of the 20 most common ngrams in fake news:")
    print(fake_news_ngram_fred.most_common(20))
    print('\n')

    # find frequencies of words
    fake_news_word_fred = word_frequency(fake_news_joined)
    print("Frequency of the 20 most common words in fake news:")
    print(fake_news_word_fred.most_common(20))
    print('\n')

    #----------------------------------------------------------------------
    # Try building a simple model
    # combine the real and fake news into one data frame
    # shuffle the dataset
    news_df = pd.concat([real_news_df, fake_news_df]).sample(frac=1, random_state=1).reset_index(drop=True)
    pack = decision_tree_model(news_df)

    # make a prediction
    # change the file path below to the absolute file path of a sample
    make_prediction(pack, "/Users/jack/programming/machine_learning/CSS581_Project_Repo/CSS581_Project/fakeNewsDatasets/fakeNewsDataset/legit/biz14.legit.txt")


    """
    #Claim dataframe
    df_claim_fake_v1 = pd.read_csv("CoAID/05-01-2020/ClaimFakeCOVID-19.csv")
    df_claim_fake_v2 = pd.read_csv("CoAID/07-01-2020/ClaimFakeCOVID-19.csv")
    df_claim_real_v1 = pd.read_csv("CoAID/05-01-2020/ClaimRealCOVID-19.csv")
    df_claim_real_v2 = pd.read_csv("CoAID/07-01-2020/ClaimRealCOVID-19.csv")

    df_claim_fake = claim_version_combine(df_claim_fake_v1, df_claim_fake_v2)
    df_claim_real = claim_version_combine(df_claim_real_v1, df_claim_real_v2)

    #news dataframe
    df_news_fake_v1 = pd.read_csv("CoAID/05-01-2020/NewsFakeCOVID-19.csv")
    df_news_fake_v2 = pd.read_csv("CoAID/07-01-2020/NewsFakeCOVID-19.csv")
    df_news_real_v1 = pd.read_csv("CoAID/05-01-2020/NewsRealCOVID-19.csv")
    df_news_real_v2 = pd.read_csv("CoAID/07-01-2020/NewsRealCOVID-19.csv")

    df_news_fake = news_version_combine(df_news_fake_v1, df_news_fake_v2)
    df_news_real = news_version_combine(df_news_real_v1, df_news_real_v2)

    #claim tweets
    df_claim_fake_tweets_v1 = pd.read_csv("CoAID/05-01-2020/ClaimFakeCOVID-19_tweets.csv")
    df_claim_fake_tweets_v2 = pd.read_csv("CoAID/07-01-2020/ClaimFakeCOVID-19_tweets.csv")
    df_claim_fake_tweets_replies_v1 = pd.read_csv("CoAID/05-01-2020/ClaimFakeCOVID-19_tweets_replies.csv")
    df_claim_fake_tweets_replies_v2 = pd.read_csv("CoAID/07-01-2020/ClaimFakeCOVID-19_tweets_replies.csv")
    df_claim_real_tweets_v1 = pd.read_csv("CoAID/05-01-2020/ClaimRealCOVID-19_tweets.csv")
    df_claim_real_tweets_v2 = pd.read_csv("CoAID/07-01-2020/ClaimRealCOVID-19_tweets.csv")
    df_claim_real_tweets_replies_v1 = pd.read_csv("CoAID/05-01-2020/ClaimRealCOVID-19_tweets_replies.csv")
    df_claim_real_tweets_replies_v2 = pd.read_csv("CoAID/07-01-2020/ClaimRealCOVID-19_tweets_replies.csv")

    df_claim_fake_tweets = tweets_version_combine(df_claim_fake_tweets_v1, df_claim_fake_tweets_v2)
    df_claim_real_tweets = tweets_version_combine(df_claim_real_tweets_v1, df_claim_real_tweets_v2)
    df_claim_fake_tweets_replies = retweets_version_combine(df_claim_fake_tweets_replies_v1, df_claim_fake_tweets_replies_v2)
    df_claim_real_tweets_replies = retweets_version_combine(df_claim_real_tweets_replies_v1, df_claim_real_tweets_replies_v2)

    #news tweets
    df_news_fake_tweets_v1 = pd.read_csv("CoAID/05-01-2020/NewsFakeCOVID-19_tweets.csv")
    df_news_fake_tweets_v2 = pd.read_csv("CoAID/07-01-2020/NewsFakeCOVID-19_tweets.csv")
    df_news_fake_tweets_replies_v1 = pd.read_csv("CoAID/05-01-2020/NewsFakeCOVID-19_tweets_replies.csv")
    df_news_fake_tweets_replies_v2 = pd.read_csv("CoAID/07-01-2020/NewsFakeCOVID-19_tweets_replies.csv")
    df_news_real_tweets_v1 = pd.read_csv("CoAID/05-01-2020/NewsRealCOVID-19_tweets.csv")
    df_news_real_tweets_v2 = pd.read_csv("CoAID/07-01-2020/NewsRealCOVID-19_tweets.csv")
    df_news_real_tweets_replies_v1 = pd.read_csv("CoAID/05-01-2020/NewsRealCOVID-19_tweets_replies.csv")
    df_news_real_tweets_replies_v2 = pd.read_csv("CoAID/07-01-2020/NewsRealCOVID-19_tweets_replies.csv")

    df_news_fake_tweets = tweets_version_combine(df_news_fake_tweets_v1, df_news_fake_tweets_v2)
    df_news_real_tweets = tweets_version_combine(df_news_real_tweets_v1, df_news_real_tweets_v2)
    df_news_fake_tweets_replies = retweets_version_combine(df_news_fake_tweets_replies_v1, df_news_fake_tweets_replies_v2)
    df_news_real_tweets_replies = retweets_version_combine(df_news_real_tweets_replies_v1, df_news_real_tweets_replies_v2)

    # merge main content with tweets
    df_claim_fake_tweets_merged = merge_main_content_tweets(df_claim_fake, df_claim_fake_tweets)
    df_claim_real_tweets_merged = merge_main_content_tweets(df_claim_real, df_claim_real_tweets)
    df_news_fake_tweets_merged = merge_main_content_tweets(df_news_fake, df_news_fake_tweets)
    df_news_real_tweets_merged = merge_main_content_tweets(df_news_real, df_news_real_tweets)

    # main content merged with tweets and retweets
    df_claim_fake_merged = merge_main_content_retweets(df_claim_fake_tweets_merged, df_claim_fake_tweets_replies)
    df_claim_real_merged = merge_main_content_retweets(df_claim_real_tweets_merged, df_claim_real_tweets_replies)
    df_news_fake_merged = merge_main_content_retweets(df_news_fake_tweets_merged, df_news_fake_tweets_replies)
    df_news_real_merged = merge_main_content_retweets(df_news_real_tweets_merged, df_news_real_tweets_replies)

    # mergin and processing claim and news main content
    df_claim = claim_preprocess(df_claim_fake_merged, df_claim_real_merged)
    df_news = news_preprocess(df_news_fake_merged, df_news_real_merged)
    
    #-------------------------------------------------------------------
    # concatenate all contents of real news
    real_news_contents = np.array(df_news_real['content'].tolist())
    real_news_joined = " "
    real_news_joined = real_news_joined.join(real_news_contents)

    # Find ngrams
    real_news_ngram_fred = ngram_frequency(real_news_joined, 4)
    print('\n')
    print("Frequency of the 20 most common ngrams in real news:")
    print(real_news_ngram_fred.most_common(20))
    print('\n')

    # find frequencies of words
    real_news_word_fred = word_frequency(real_news_joined)
    print("Frequency of the 20 most common words in real news:")
    print(real_news_word_fred.most_common(20))
    print('\n')

    #--------------------------------------------------------------------
    # concatenate all contents of fake news
    fake_news_contents = np.array(df_news_fake['content'].tolist())
    fake_news_joined = " "
    fake_news_joined = fake_news_joined.join(fake_news_contents)
    print(fake_news_joined)

    # Find ngrams
    fake_news_ngram_fred = ngram_frequency(fake_news_joined, 4)
    print('\n')
    print("Frequency of the 20 most common ngrams in fake news:")
    print(fake_news_ngram_fred.most_common(20))
    print('\n')

    # find frequencies of words
    fake_news_word_fred = word_frequency(fake_news_joined)
    print("Frequency of the 20 most common words in fake news:")
    print(fake_news_word_fred.most_common(20))
    print('\n')
    """

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
