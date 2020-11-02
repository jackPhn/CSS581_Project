import pandas as pd
import os


def get_dataset_from_file(dataset: str, is_fake: bool):
    """
    Return a data frame from files
    :param dataset: name of the dataset
    :param is_fake: whether fake news
    :return: a data frame of the dataset
    """
    # list of news titles
    titles = []
    # list of news contents
    contents = []

    quality = 'fake' if is_fake else 'legit'

    # grab data from celebrity dataset
    cwd = os.getcwd()
    fakeCelebDir = os.path.join(os.path.join(os.path.join(cwd, 'fakeNewsDatasets'), dataset), quality)

    for filename in os.listdir(fakeCelebDir):

        with open(os.path.join(fakeCelebDir, filename), 'r') as file:
            # read and store the title
            title = file.readline()
            titles.append(title)

            # read and store the content
            content_lines = file.readlines()
            content = " ".join(content_lines)
            contents.append(content)

    newsDf = pd.DataFrame({'Title': titles, 'Content': contents})
    if is_fake:
        newsDf['Is fake'] = 1
    else:
        newsDf['Is fake'] = 0

    return newsDf


def build_real_news_dataframe(include_celeb: bool = False):
    """
    Build real news data frame
    :param include_celeb: whether or nor to include the celebrity dataset
    :return: the real news data frame
    """
    # real news dataset
    column_names = ['Title', 'Content', 'Is fake']
    realDf = pd.DataFrame(columns=column_names)

    # build the dataframe with the non-celeb news
    realDf = realDf.append(get_dataset_from_file("fakeNewsDataset", False), ignore_index=True)

    if include_celeb:
        realDf = realDf.append(get_dataset_from_file("celebrityDataset", False), ignore_index=True)

    return realDf


def build_fake_news_dataframe(include_celeb: bool = False):
    """
    Build fake news data frame
    :param include_celeb: whether or not to include the celebrity dataset
    :return: the fake news data frame
    """
    # fake new dataset
    column_names = ['Title', 'Content', 'Is fake']
    fakeDf = pd.DataFrame(columns=column_names)

    # build the data frame with the non-celeb news
    fakeDf = fakeDf.append(get_dataset_from_file("fakeNewsDataset", True), ignore_index=True)

    if include_celeb:
        fakeDf = fakeDf.append(get_dataset_from_file("celebrityDataset", True), ignore_index=True)

    return fakeDf
