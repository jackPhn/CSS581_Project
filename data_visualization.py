import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import nltk
import plotly.express as px
from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    )

def visualize_real_fake(news_df):
    # displaying ligit and fake count
    news_md_df = news_df.copy()
    news_md_df['is_fake'] = pd.Series(np.where(news_md_df.is_fake.values == 1, True, False),
              news_md_df.index)
    is_fake_data = news_md_df[(news_md_df.is_fake == True)][['is_fake']]
    is_no_fake_data = news_md_df[(news_md_df.is_fake == False)][['is_fake']]
    plotSingleHistogram(is_fake_data, is_no_fake_data, "Is Fake", "Count", "Ligit and Fake Count")

def visualize_news_celebrity(news_df):
    # displaying news and celebrity count
    news_md_df = news_df.copy()
    news_md_df['is_fake'] = pd.Series(np.where(news_md_df.is_news.values == 1, True, False),
                                      news_md_df.index)
    is_news_data = news_md_df[(news_md_df.is_news == True)][['is_news']]
    is_no_news_data = news_md_df[(news_md_df.is_news == False)][['is_news']]
    # bins = compute_histogram_bins(is_news_data, 10)
    plotSingleHistogram(is_news_data, is_no_news_data, "Is News", "Count", "News and Celebroty Count")

# visualize content catagory
# def visulaize_content_catagory(df):
#     plt.figure(figsize = (8,8))
#     sns.countplot(y="subject", data= df)
#     plt.show()

# visualis fake and legit
def visulaize_fake_legit(df):
    plt.figure(figsize = (8,8))
    sns.countplot(y="is_fake", data= df)
    plt.show()

def visualize_fake_word_cloud_plot(df, stop_words):
    plt.figure(figsize=(12,12))
    wc = WordCloud(max_words = 2000, width = 8000, height = 8000, stopwords = stop_words).generate(" ".join(df[df.is_fake==1].clean_joined))
    plt.imshow(wc, interpolation='bilinear')
    plt.show()

def visualize_legit_word_cloud_plot(df, stop_words):
    plt.figure(figsize=(12,12))
    wc = WordCloud(max_words = 2000, width = 8000, height = 8000, stopwords = stop_words).generate(" ".join(df[df.is_fake==0].clean_joined))
    plt.imshow(wc, interpolation='bilinear')
    plt.show()

def visualize_word_distribution(df):
    fig = px.histogram(x=[len(nltk.wordtokenize(x)) for x in df.clean_joined], nbins=100)
    fig.show()

def visualize_confusion_matrix(prediction, y_test):
    '''
    :param prediction:
    :param y_test:
    :return:
    '''
    cm = confusion_matrix(list(y_test), prediction)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,annot=True)
    plt.show()


#plot single bar histogram
def plotSingleHistogram(first_bar_data, second_bar_data, xlable, ylable, title):
    bars = plt.bar([1,2],[len(first_bar_data), len(second_bar_data)])
    bars[1].set_color('green')
    lablelist = [0, 'True', 'False']
    tickvalues = range(0, len(lablelist))
    plt.xticks(ticks = tickvalues ,labels = lablelist, rotation='horizontal')
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.title(title)
    plt.show()





# #plot dual bar histogram for any type of feature
# def plotGeneralHistogram(group, first_bar_data, second_bar_data, first_bar_name, second_bar_name, title, ylable, xlable, x_tick_lable_one, x_tick_lable_two):
#
#     ind = np.arange(group)  # the x locations for the groups
#     width = 0.35       # the width of the bars
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(ind, first_bar_data, width)
#     rects2 = ax.bar(ind + width, second_bar_data, width)
#
#     # add some text for labels, title and axes ticks
#     ax.set_title(title)
#     ax.set_xticks(ind + width / 2)
#     ax.set_ylabel(ylable)
#     ax.set_xticklabels((x_tick_lable_one, x_tick_lable_two))
#     ax.legend((rects1[0], rects2[0]), (first_bar_name, second_bar_name))
#     autolabel(rects1, ax)
#     autolabel(rects2, ax)
#     # plt.rcParams["figure.figsize"] = [7,6]
#     # plt.figure(figsize=(7,7))
#     plt.xlabel(xlable)
#     plt.show()

#
# #write the value of the bar on top
# def autolabel(rects, ax):
#     """
#     Attach a text label above each bar displaying its height
#     """
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 '%d' % int(height),
#                 ha='center', va='bottom')
