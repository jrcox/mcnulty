import pandas as pd
import seaborn as sns
import pickle
import pandas as pd
%matplotlib inline
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
pd.set_option('display.float_format', lambda x: '%.3f' % x)
sns.set_style("white")

#open dataframe
with open('masterdf.pkl', 'rb') as masterdfpkl:
    masterdf= pickle.load(masterdfpkl)
    
masterdf = masterdf[masterdf.ruling != 'Full Flop']
masterdf = masterdf[masterdf.ruling != 'Half Flip']
masterdf = masterdf[masterdf.ruling != 'No Flip']

masterdf.replace('governor', 'Governor', inplace = True)
masterdf.replace('Senate', 'Senator', inplace = True)

dummies = pd.get_dummies(masterdf['state'])
dummies2 = pd.get_dummies(masterdf['party'])
masterdf = pd.concat([masterdf, dummies2, dummies], axis=1)      

party_ = masterdf.groupby(['ruling', 'party'], as_index=False)
chamber_ = masterdf.groupby(['chamber','ruling', 'party'])
state_ = masterdf.groupby(['state', 'ruling'], as_index = False)

#create dataframe for D3 viz
state_by_ruling = state_.pivot(index='state', columns='ruling', values='count')
cols = ['True', 'Mostly True', 'Half-True', 'Mostly False', 'False', 'Pants on Fire!']
state_by_ruling = state_by_ruling.ix[:, cols]

state_by_ruling.fillna(value=0, inplace = True)
state_by_ruling['percentage'] = (state_by_ruling.sum(axis=1)/419)*100

state_by_ruling['true_count'] = state_by_ruling['True'] + state_by_ruling['Mostly True']
state_by_ruling['false_count'] = state_by_ruling['False'] + state_by_ruling['Mostly False'] + state_by_ruling['Half-True'] + state_by_ruling['Pants on Fire!']
state_by_ruling['total_count'] = state_by_ruling['true_count'] + state_by_ruling['false_count']
state_by_ruling['percent_true'] = (state_by_ruling['true_count']/state_by_ruling['total_count'])*100
state_by_ruling['percent_false'] = (state_by_ruling['false_count']/state_by_ruling['total_count'])*100

columns_csv = ['True', 'Mostly True', 'Half-True', 'Mostly False', 'False', 'Pants on Fire!',
          'percentage', 'percent_true']
state_by_ruling_csv = state_by_ruling[columns_csv]

state_by_ruling_csv = state_by_ruling_csv.astype(int)

#------
#making distribution histrogram by % frequency
ncount = len(masterdf)

plt.figure(figsize=(12,8))
ax = sns.countplot(x="ruling", data=masterdf, order=['True', 'Mostly True', 'Half-True', 'Mostly False',
                            'False', 'Pants on Fire!'])
plt.title('Distribution of Statements')
plt.xlabel('Ruling')

# Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Frequency [%]')

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text

# Fix the frequency range to 0-100
ax2.set_ylim(0,30)
ax.set_ylim(0,100)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)
#------

#------
#Make stacked histogram of republican vs democrats
#Read in data & create total column
masterdf["total"] = masterdf.Republican + masterdf.Democrat
repubdf = masterdf.loc[masterdf['party'] == "Republican"]
democratdf = masterdf.loc[masterdf['party'] == "Democrat"]

#Set general plot properties
sns.set_style("white")
sns.set_context({"figure.figsize": (24, 10)})

#Plot 1 - background - "total" (top) series
top_plot = sns.countplot(x = repubdf.ruling,  order = ['True', 'Mostly True', 'Half-True', 'Mostly False',
                            'False', 'Pants on Fire!'], color = "#EC7063")

#Plot 2 - overlay - "bottom" series
bottom_plot = sns.countplot(x = democratdf.ruling, order = ['True', 'Mostly True', 'Half-True', 'Mostly False',
                            'False', 'Pants on Fire!'], color = "#33bbff" )


# topbar = plt.Rectangle((0,0),1,1)
# bottombar = plt.Rectangle((0,0),1,1)
# l = plt.legend([bottombar, topbar], ['Republicans', 'Democrats'], loc=1, ncol = 2, prop={'size':16})
# l.draw_frame(False)

#Optional code - Make plot look nicer
sns.despine(left=True)
bottom_plot.set_ylabel("Count")
bottom_plot.set_xlabel("Ruling")

#Set fonts to consistent 16pt size
for item in ([bottom_plot.xaxis.label,bottom_plot.yaxis.label] +
             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
    item.set_fontsize(20)
#------


#------
#Stacked histograms for senate vs governors
#Read in data & create total column
masterdf["total"] = masterdf.Republican + masterdf.Democrat
sendf = masterdf.loc[masterdf['chamber'] == "Senator"]
govdf = masterdf.loc[masterdf['chamber'] == "Governor"]

#Set general plot properties
sns.set_style("white")
sns.set_context({"figure.figsize": (24, 10)})

#Plot 1 - background - "total" (top) series
top_plot = sns.countplot(x = sendf.ruling,  order = ['True', 'Mostly True', 'Half-True', 'Mostly False',
                            'False', 'Pants on Fire!'], color = "#7eb1b7")

#Plot 2 - overlay - "bottom" series
bottom_plot = sns.countplot(x = govdf.ruling, order = ['True', 'Mostly True', 'Half-True', 'Mostly False',
                            'False', 'Pants on Fire!'], color = "#d2e3ae" )


# topbar = plt.Rectangle((0,0),1,1)
# bottombar = plt.Rectangle((0,0),1,1)
# l = plt.legend([bottombar, topbar], ['Republicans', 'Democrats'], loc=1, ncol = 2, prop={'size':16})
# l.draw_frame(False)

#Optional code - Make plot look nicer
sns.despine(left=True)
bottom_plot.set_ylabel("Count")
bottom_plot.set_xlabel("Ruling")
#bottom_plot.legend("Governors", "Senators")
bottom_plot.legend(loc='upper left')


#Set fonts to consistent 16pt size
for item in ([bottom_plot.xaxis.label,bottom_plot.yaxis.label] +
             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
    item.set_fontsize(20)    
#------

#------
#make plot of frequency of statement for entire party
grouped = masterdf.groupby(['party'], sort=False)
ruling_counts = grouped['ruling'].value_counts(normalize=True, sort=False)

ruling_data = [
    {'party': party, 'ruling': ruling, 'percentage': percentage*100} for 
    (party, ruling), percentage in dict(ruling_counts).items()
]

df_ruling = pd.DataFrame(ruling_data)

p = sns.barplot(x="party", y="percentage", hue="ruling", 
                hue_order = ['True', 'Mostly True', 'Half-True', 'Mostly False',
                            'False', 'Pants on Fire!'],
                data=df_ruling)
plt.ylabel('Percentage of party statements')
plt.xlabel('Party')
_ = plt.setp(p.get_xticklabels(), rotation=0)
#------


#------
#make plot of frequency of statement for chamber type
grouped = masterdf.groupby(['chamber'], sort=False)
ruling_counts = grouped['ruling'].value_counts(normalize=True, sort=False)

ruling_data = [
    {'chamber': chamber, 'ruling': ruling, 'percentage': percentage*100} for 
    (chamber, ruling), percentage in dict(ruling_counts).items()
]

df_ruling = pd.DataFrame(ruling_data)

p = sns.barplot(x="chamber", y="percentage", hue="ruling", 
                hue_order = ['True', 'Mostly True', 'Half-True', 'Mostly False',
                            'False', 'Pants on Fire!'],
                data=df_ruling)
_ = plt.setp(p.get_xticklabels(), rotation=0)
#------

