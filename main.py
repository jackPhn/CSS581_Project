import numpy as np
import pandas as pd

from data_compilation import (
    build_real_news_dataframe,
    build_fake_news_dataframe
)

from data_exploration import (
    validate_null_value,
    validate_unique_recored,
    ngram_frequency,
    word_frequency
)
from model_building import (
    classical_models,
    make_prediction,
    deep_learning_model,
    create_pad_sequence,
    predict_stml_model,
    build_ltsm_model
)
from hyperparameter_tuning import (
    none_dl_grid_search,
    dl_grid_search

)


from data_visualization import (
    visualize_real_feak,
    visualize_news_celebrity,
    visualize_fake_word_cloud_plot,
    visulaize_fake_ligit,
    visualize_ligit_word_cloud_plot,
    visualize_word_distribution,
    visualize_confusion_matrix,
)

from feature_engineering import(
    process_feature_engineering
)


def main():
    # data loading
    real_news_df = build_real_news_dataframe(include_celeb=True)
    fake_news_df = build_fake_news_dataframe(include_celeb=True)
    news_df = pd.concat([real_news_df, fake_news_df])

    # find the longest news content
    news_length = [len(news) for news in news_df['Content'].tolist()]
    print("The number of words in the longest piece of news is:", np.max(news_length))
    print()

    # find out the total number of unique words in the corpus
    news_contents = np.array(news_df['Content'].tolist())
    news_df_joined = " ".join(news_contents).lower()
    unique_words = list(word_frequency(news_df_joined).keys())
    print("The number of unique words in the news contents is:", len(unique_words))
    print()

    # find missing values
    validate_null_value(news_df)
    validate_unique_recored(news_df)

    # -------------------------------------------------------------------------------------------------
    # concatenate all contents of real news
    real_news_contents = np.array(real_news_df['Content'].tolist())
    real_news_joined = " ".join(real_news_contents)

    # Find ngrams
    real_news_ngram_fred = ngram_frequency(real_news_joined, 3)
    print()
    print("Frequency of the 20 most common ngrams in real news:")
    print(real_news_ngram_fred.most_common(20))

    # find frequencies of words
    real_news_word_fred = word_frequency(real_news_joined)
    print()
    print("Frequency of the 20 most common words in real news:")
    print(real_news_word_fred.most_common(20))

    # --------------------------------------------------------------------
    # concatenate all contents of fake news
    fake_news_contents = np.array(fake_news_df['Content'].tolist())
    fake_news_joined = " ".join(fake_news_contents)

    # Find ngrams
    fake_news_ngram_fred = ngram_frequency(fake_news_joined, 3)
    print()
    print("Frequency of the 20 most common ngrams in fake news:")
    print(fake_news_ngram_fred.most_common(20))

    # find frequencies of words
    fake_news_word_fred = word_frequency(fake_news_joined)
    print()
    print("Frequency of the 20 most common words in fake news:")
    print(fake_news_word_fred.most_common(20))
    print()

    # -------------------------------------------------------------------------------------------------
    # Visualization
    # visualize_real_feak(news_df)
    # visualize_news_celebrity(news_df)
    # visulaize_fake_ligit(news_df)
    # -------------------------------------------------

    # LSTM and RNN
    # df_clean, stop_words = process_feature_enginering(news_df)
    # visualize_fake_word_cloud_plot(df_clean, stop_words)
    # visualize_ligit_word_cloud_plot(df_clean, stop_words)

    # -------------------------------------------------------------------------------------------------
    # classical models
    # shuffle the dataset
    # This part will take a significant amount of time to run, comment out if not needed
    news_df = news_df.sample(frac=1, random_state=1).reset_index(drop=True)

    # using grid search to find the best parameters for several models
    # print()
    # print("Performing grid search for non-deep-learning models")
    # none_dl_grid_search(news_df)

    """
    # cross validation and training
    print("Performing cross validation and training models")
    classic_pack = classical_models(news_df)

    # change the file path below to the absolute file path of a sample used to make prediction on
    sample_file_path = "/Users/jack/programming/machine_learning/CSS581_Project_Repo/CSS581_Project/fakeNewsDatasets/fakeNewsDataset/fake/tech008.fake.txt"
    # make a prediction
    print()
    model_name = "Logistic Regression"
    print("Making a prediction with", model_name)
    make_prediction(classic_pack, sample_file_path, model_name)

    # -------------------------------------------------------------------------------------------------
    # deep learning model with word embedding
    print()
    print("Evaluating and training a deep learning model")
    dl_pack = deep_learning_model(news_df)
    # make a prediction
    # change the file path below to the absolute file path of a sample
    print()
    print("Making a prediction with the deep learning model")
    make_prediction(dl_pack, sample_file_path, "dl")
    """
    # hyperparameter tuning for deep learning model
    dl_grid_search(news_df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
