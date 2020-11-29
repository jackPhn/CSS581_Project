import numpy as np
import pandas as pd

from data_compilation import (
    build_real_news_dataframe,
    build_fake_news_dataframe,
    clean_text
)

from data_exploration import (
    validate_null_value,
    validate_unique_record,
    ngram_frequency,
    word_frequency,
    visualize_composition
)
from model_building import (
    classical_models,
    make_prediction,
    deep_learning_model,
    create_pad_sequence,
    predict_lstm_model,
    build_lstm_model,
    create_lstm_predictive_model
)
from hyperparameter_tuning import (
    none_dl_grid_search,
    dl_grid_search
)

from data_visualization import (
    visualize_real_fake,
    visualize_news_celebrity,
    visualize_fake_word_cloud_plot,
    visulaize_fake_legit,
    visualize_legit_word_cloud_plot,
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
    # Clean the data by removing punctuation
    news_df = clean_text(news_df, "Content", "Content")
    news_df = clean_text(news_df, "Title", "Title")

    # -------------------------------------------------------------------------------------------------
    # Data exploration
    response = input("Do you want to perform data exploration? Enter (Y)es or (N)o").lower()
    if response == 'y':
        # find the longest news content
        max_content_length = np.max([len(news) for news in news_df['Content'].tolist()])
        print("The number of words in the longest piece of news is:", max_content_length)
        print()

        # find out the total number of unique words in the corpus
        news_contents = np.array(news_df['Content'].tolist())
        news_df_joined = " ".join(news_contents).lower()
        unique_words = list(word_frequency(news_df_joined).keys())
        print("The number of unique words in the news contents is:", len(unique_words))
        print()

        # find missing values
        validate_null_value(news_df)
        validate_unique_record(news_df)

        # ---------------------------------------------------------------------------------------------
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

        # ---------------------------------------------------------------------------------------------
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

        # ---------------------------------------------------------------------------------------------
        # Data visualization
        visualize_composition(fake_news_df, real_news_df)


        # ---------------------------------------------------------------------------------------------
        # Visualization
        #visualize_real_fake(news_df)
        #visualize_news_celebrity(news_df)
        #visulaize_fake_legit(news_df)


    # LSTM and RNN
    # create_lstm_predictive_model(news_df)

    # -------------------------------------------------------------------------------------------------
    # shuffle the dataset
    news_df = news_df.sample(frac=1, random_state=1).reset_index(drop=True)

    response = input("Do you want to perform cross validation (v) or hyperparameter tuning (t)?").lower()

    while True:
        if response == 'v':
            is_dl = input("Deep learning? (Y or N)").lower()

            if is_dl == 'n':
                # cross validation and training for classical models
                print("Performing cross validation and training models")
                classic_pack = classical_models(news_df)

            else:
                # deep learning model with word embedding
                print("Evaluating and training a deep learning model")
                dl_pack = deep_learning_model(news_df)

            # change the file path below to the absolute file path of a sample used to make prediction on
            # sample_file_path = "/Users/jack/programming/machine_learning/CSS581_Project_Repo/CSS581_Project/fakeNewsDatasets/fakeNewsDataset/fake/tech008.fake.txt"

            # make a prediction
            # model_name = "Logistic Regression"
            # print("Making a prediction with", model_name)
            # make_prediction(classic_pack, sample_file_path, model_name)

            # make a prediction
            #print("Making a prediction with the deep learning model")
            #make_prediction(dl_pack, sample_file_path, "dl")

        elif response == 't':
            is_dl = input("Deep learning? (Y or N)").lower()

            if is_dl == 'n':
                #using grid search to find the best parameters for several models
                print("Performing grid search for non-deep-learning models")
                none_dl_grid_search(news_df)

            else:
                # hyperparameter tuning for deep learning model
                print("Performing grid search for deep learning model")
                dl_grid_search(news_df)

        is_end = input("Do you want to continue? (Y or N)").lower()
        if is_end == 'n':
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
