import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
from tabulate import tabulate

from data_compilation import (
    build_real_news_dataframe,
    build_fake_news_dataframe,
    clean_text,
    remove_stop_words
)

from data_exploration import (
    validate_null_value,
    ngram_frequency,
    word_frequency,
    visualize_composition,
    count_unique_words,
    find_longest_sequence_length
)
from model_building import (
    classical_models,
    make_prediction,
    deep_learning_model,
)
from hyperparameter_tuning import (
    none_dl_grid_search,
    dl_grid_search
)


def build_validate_and_tune(news_df):
    """
    Helper function to build models, validate, and tune hyperparameters
    :param news_df: news dataframe
    :return: None
    """
    # as the user for a specific action
    response = input(
        "Do you want to perform cross validation (v) or hyperparameter tuning (t)? Press any other keys to skip.").lower()

    # build models and validate
    if response == 'v':
        is_dl = input("Deep learning? (Y or N)").lower()

        if is_dl == 'n':
            # cross validation and training for classical models
            print("Performing cross validation and training on classical machine learning models")
            classical_models(news_df)

        else:
            # deep learning model with word embedding
            print("Evaluating and training a deep learning model")
            deep_learning_model(news_df)

    # tune hyperparameters
    elif response == 't':
        is_dl = input("Deep learning? (Y or N)").lower()

        if is_dl == 'n':
            # using grid search to find the best parameters for several classical models
            print("Performing grid search for classical machine learning models")
            none_dl_grid_search(news_df)

        else:
            # hyperparameter tuning for deep learning model
            print("Performing grid search for deep learning model")
            dl_grid_search(news_df)


def predict_single_case():
    """
    Helper function to make a prediction on an article using saved models
    :return: None
    """
    # check the output folder for saved models
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, 'output')
    # names of files that store saved models
    model_files = []
    # names of saved models
    model_names = []
    # extension of a model files
    model_extensions = []
    for fname in os.listdir(output_dir):
        if fname.endswith('_Model.pkl') or fname.endswith('.h5'):
            model_files.append(fname)
            name = fname.split('.')
            model_names.append(name[0])
            model_extensions.append(name[1])

    # ask if the user want to make a prediction if there are saved models
    want_to_predict = input("Do you want to make a prediction? (Y)es or (N)o?").lower()
    confirm_predict = True if want_to_predict == 'y' else False

    # prompt model select
    if confirm_predict:
        print("Available models are:", model_names)
        model_index = int(input("Enter the 0-based index of a model in the list above to select: "))

        if model_extensions[model_index] == 'pkl':
            # load the classical model
            with open(os.path.join(cwd, "output/" + model_files[model_index]), 'rb') as infile:
                model = pickle.load(infile)

            # load the feature transformers
            none_dl_transformers_fname = 'none_dl_input_transformers.pkl'
            with open(os.path.join(cwd, "output/" + none_dl_transformers_fname), 'rb') as infile:
                cv_ngram, tfidf_content, tfidf_title = pickle.load(infile)
            input_transformers = {'cv_ngram': cv_ngram, 'tfidf_content': tfidf_content, 'tfidf_title': tfidf_title}

            is_dl = False

        else:
            # load the deep learning model
            model = tf.keras.models.load_model(os.path.join(cwd, "output/" + model_files[model_index]))

            # load the tokenizer
            dl_tokenizer_fname = 'trained_tokenizer.pkl'
            with open(os.path.join(cwd, "output/" + dl_tokenizer_fname), 'rb') as infile:
                trained_tokenizer = pickle.load(infile)

            input_transformers = {'tokenizer': trained_tokenizer}

            # load the embedding parameters
            with open(os.path.join(cwd, "output/" + "embedding_dims.txt"), 'r') as infile:
                line = infile.readline()
                values = line.split(' ')
                input_transformers['vocab_size'] = int(values[0])
                input_transformers['embedding_dim'] = int(values[1])
                input_transformers['max_length'] = int(values[2])

            is_dl = True

        # prompt for a file
        input_path = input("Enter the absolute path to the input file: ")

        # make a prediction
        print("Making a prediction with", model_names[model_index])
        make_prediction(model, input_transformers, input_path, is_dl)
        print()


def main():
    # data loading
    real_news_df = build_real_news_dataframe(include_celeb=True)
    fake_news_df = build_fake_news_dataframe(include_celeb=True)
    news_df = pd.concat([real_news_df, fake_news_df])

    # Clean the data by removing punctuation
    processed_real_news_df = clean_text(real_news_df, 'Content', 'Content')
    processed_real_news_df = clean_text(real_news_df, 'Title', 'Title')
    processed_fake_news_df = clean_text(fake_news_df, 'Content', 'Content')
    processed_fake_news_df = clean_text(fake_news_df, 'Title', 'Title')

    # join the process dataframe
    processed_news_df = pd.concat([processed_real_news_df, processed_fake_news_df])

    # remove stop words from the titles
    processed_news_df = remove_stop_words(processed_news_df, 'Title')
    processed_news_df = remove_stop_words(processed_news_df, 'Content')

    # -------------------------------------------------------------------------------------------------
    # Data exploration
    response = input("Do you want to perform data exploration? Enter (Y)es or (N)o").lower()
    if response == 'y':
        # display the dataset
        print(tabulate(processed_news_df.head(5), headers='keys', tablefmt='psql'))
        print(processed_news_df.shape)

        # find missing values
        validate_null_value(news_df)

        # ---------------------------------------------------------------------------------------------
        # explore the data preprocessing
        # find the longest news content
        print("Attributes preprocessing:")
        max_content_length = find_longest_sequence_length(news_df['Content'].tolist())
        print("The number of words in the longest piece of news is:", max_content_length)
        print()

        # find out the total number of unique words in the news contents
        content_num_unique_words = count_unique_words(news_df['Content'].tolist())
        print("The number of unique words in the news contents is:", len(content_num_unique_words))
        print()

        # find the longest headline
        max_headline_length = find_longest_sequence_length(news_df['Title'].tolist())
        print("The number of words in the longest headline is:", max_headline_length)
        print()

        # find out the total number of unique words in the headlines
        headline_num_unique_words = count_unique_words(news_df['Title'].tolist())
        print("The number of unique words in the news headlines is:", len(headline_num_unique_words))
        print()

        # ---------------------------------------------------------------------------------------------
        # explore the data postprocessing
        print("Attributes postprocessing:")
        # find the longest news content
        max_content_length = find_longest_sequence_length(processed_news_df['Processed Content'].tolist())
        print("The number of words in the longest piece of news is:", max_content_length)
        print()

        # find out the total number of unique words in the news contents
        content_num_unique_words = count_unique_words(processed_news_df['Processed Content'].tolist())
        print("The number of unique words in the news contents is:", len(content_num_unique_words))
        print()

        # find the longest headline
        max_headline_length = find_longest_sequence_length(processed_news_df['Processed Title'].tolist())
        print("The number of words in the longest headline is:", max_headline_length)
        print()

        # find out the total number of unique words in the headlines
        headline_num_unique_words = count_unique_words(processed_news_df['Processed Title'].tolist())
        print("The number of unique words in the news headlines is:", len(headline_num_unique_words))
        print()

        # ---------------------------------------------------------------------------------------------
        # concatenate all contents of real news
        real_news_contents = np.array(processed_real_news_df['Content'].tolist())
        real_news_joined = " ".join(real_news_contents)

        # Find ngrams frequency
        real_news_3gram_freq = ngram_frequency(real_news_joined, 3)
        print()
        print("Frequency of the 20 most common 3-grams in real news:")
        print(real_news_3gram_freq.most_common(20))
        print()
        real_news_4gram_freq = ngram_frequency(real_news_joined, 4)
        print("Frequency of the 20 most common 4-grams in real news:")
        print(real_news_4gram_freq.most_common(20))

        # find frequencies of words
        real_news_word_freq = word_frequency(real_news_joined)
        print()
        print("Frequency of the 20 most common words in real news:")
        print(real_news_word_freq.most_common(20))

        # ---------------------------------------------------------------------------------------------
        # concatenate all contents of fake news
        fake_news_contents = np.array(processed_fake_news_df['Content'].tolist())
        fake_news_joined = " ".join(fake_news_contents)

        # Find ngrams frequency
        fake_news_3gram_freq = ngram_frequency(fake_news_joined, 3)
        print()
        print("Frequency of the 20 most common 3-grams in fake news:")
        print(fake_news_3gram_freq.most_common(20))
        fake_news_4gram_freq = ngram_frequency(fake_news_joined, 4)
        print()
        print("Frequency of the 20 most common 4-grams in fake news:")
        print(fake_news_4gram_freq.most_common(20))

        # find frequencies of words
        fake_news_word_freq = word_frequency(fake_news_joined)
        print()
        print("Frequency of the 20 most common words in fake news:")
        print(fake_news_word_freq.most_common(20))
        print()

        # ---------------------------------------------------------------------------------------------
        # Data visualization
        visualize_composition(news_df)

    # -------------------------------------------------------------------------------------------------
    # shuffle the dataset
    processed_news_df = processed_news_df.sample(frac=1, random_state=1).reset_index(drop=True)

    # start execution loop
    run_again = True
    while run_again:

        # build models, validate, and tune hyperparameters
        build_validate_and_tune(processed_news_df)

        # make a prediction on a news article
        predict_single_case()

        # ask if the user wants to execute again
        is_end = input("Do you want to execute again? (Y or N)").lower()
        if is_end == 'n':
            run_again = False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
