import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    )

def visualize_real_feak(news_df):
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

# def compute_histogram_bins(data, desired_bin_size):
#     min_val = np.min(data)
#     max_val = np.max(data)
#     min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
#     max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
#     n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
#     bins = np.linspace(min_boundary, max_boundary, n_bins)
#     return bins






#plot dual bar histogram for any type of feature
def plotGeneralHistogram(group, first_bar_data, second_bar_data, first_bar_name, second_bar_name, title, ylable, xlable, x_tick_lable_one, x_tick_lable_two):

    ind = np.arange(group)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, first_bar_data, width)
    rects2 = ax.bar(ind + width, second_bar_data, width)

    # add some text for labels, title and axes ticks
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_ylabel(ylable)
    ax.set_xticklabels((x_tick_lable_one, x_tick_lable_two))
    ax.legend((rects1[0], rects2[0]), (first_bar_name, second_bar_name))
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    # plt.rcParams["figure.figsize"] = [7,6]
    # plt.figure(figsize=(7,7))
    plt.xlabel(xlable)
    plt.show()


#plot dual bar histogram age againest any feature
def plotAgeHistogram(group, first_bar_data, second_bar_data, first_bar_name, seconed_bar_name, title, xlable):

    # show_data = dataset[(dataset.Show=="Yes")]
    # notshow_data = dataset[(dataset.Show=="No")]

    age = ['{}-{}'.format(i*10, (i+1)*10) for i in range(10)]
    age_bins = [i*10 for i in range(11)] # = [0,10,20,30,40,50,60,70,80,90,100]

    show_hist=np.histogram(first_bar_data[xlable],bins=age_bins,range=(0,100))
    not_show_hist=np.histogram(second_bar_data[xlable],bins=age_bins,range=(0,100))

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.bar(np.arange(group)-0.15, show_hist[0], width=0.1, label=first_bar_name)
    ax.bar(np.arange(group)+0.1, not_show_hist[0], width=0.1, label=seconed_bar_name)
    ax.set_xticks(np.arange(group))
    ax.set_xticklabels(age)
    ax.legend()
    ax.set_xlim(-0.5,9.5)
    plt.xlabel(xlable)
    plt.show()


#write the value of the bar on top
def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')




