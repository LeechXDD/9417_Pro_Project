#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install tensorflow_addons')
get_ipython().system('pip install tensorflow_decision_forests')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install tensorflow --upgrade')
get_ipython().system('pip install keras --upgrade')

# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import tensorflow as tf
#import tensorflow_decision_forests as tfdf


# In[3]:


# Loading in Dataset
#path = '/Users/benedictachun/desktop/comp9417/9417-Project/predict-student-performance-from-game-play'
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[6]:


# Load in labels for training dataset
labels = pd.read_csv('train_labels.csv')
labels['session'] = labels.session_id.apply(lambda x: int(x.split('_')[0]) )
labels['q'] = labels.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )


# ## Data Preprocessing

# #### Reducing memory storage
# This function reduces the size of the properties of the dataset to make the dataset smaller without losing information
# - Reference: https://www.kaggle.com/code/mohammad2012191/reduce-memory-usage-2gb-780mb

# In[7]:


def get_minimal_dtype(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                    
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage became: ",mem_usg," MB")
    
    return df
    


# In[8]:


train_data.info()


# In[9]:


train_data = get_minimal_dtype(train_data)


# In[10]:


train_data.info()


# #### Data Cleaning
# The columns with missing values are:
# 
# - page: This is only for notebook-related events. The missing values could indicate that the event is not related to the notebook. We could fill missing values with a placeholder like -1 to denote 'Not Applicable'.
# - room_coor_x, room_coor_y, screen_coor_x, screen_coor_y: These are the coordinates of the click, and are only relevant for click events. Similar to 'page', we could fill missing values with a placeholder.
# - hover_duration: This is only for hover events. We can use the same approach as for the coordinates.
# 

# In[11]:


# Find out columns with missing values
missing_values = train_data.isnull().sum()

# Fill missing values
train_data['page'].fillna(-1, inplace=True)
train_data['room_coor_x'].fillna(-1, inplace=True)
train_data['room_coor_y'].fillna(-1, inplace=True)
train_data['screen_coor_x'].fillna(-1, inplace=True)
train_data['screen_coor_y'].fillna(-1, inplace=True)
train_data['hover_duration'].fillna(-1, inplace=True)


# ### Exploratory Data Analysis

# ### Univariate Analysis
# 

# In[12]:


# Summary statistics
train_data.describe()


# In[13]:


test_data.describe()


# In[14]:


# Create an empty DataFrame to store the correlations for each question
correlations_by_q = pd.DataFrame()

# Iterate over each question number
for q_no in range(1, 19):
    # Select the level group for the question based on the q_no
    if q_no <= 4:
        grp = '0-4'
    elif q_no <= 12:
        grp = '5-12'
    else:
        grp = '13-22'

    # Filter the rows in the datasets based on the selected level group
    filtered_data = train_data[train_data.level_group == grp]

    # Select the labels for the related q_no
    filtered_labels = labels[labels.q == q_no].set_index('session').loc[filtered_data.reset_index().session_id]

    # Add the label to the filtered datasets
    filtered_data = filtered_data.reset_index()
    filtered_data['correct'] = filtered_labels['correct'].values

    # Calculate the correlation between each feature and the target variable
    correlations = filtered_data.drop(['session_id', 'level_group'], axis=1).corr()['correct'].sort_values(ascending=False)

    # Store the correlations in the correlations_by_q DataFrame
    correlations_by_q[q_no] = correlations


# In[15]:


correlations


# In[16]:


# Set plot style
sns.set_style("whitegrid")

# Create a function for easy plotting
def plot_count(train_data, column, title, color, rotation=0):
    plt.figure(figsize=(12,6))
    sns.countplot(data=train_data, x=column, order=train_data[column].value_counts().index, color=color)
    plt.title(title, size=16)
    plt.xticks(rotation=rotation)
    plt.show()


# #### Distribution of the Event Names
# The most common event in the dataset is 'navigate_click', followed by 'notification_click'. These events likely relate to key interactions within the game and could be influential in a model's ability to predict student performance.

# In[17]:


# Plot the distribution of event names
plot_count(train_data, 'event_name', 'Distribution of Event Names', 'skyblue')


# #### Distribution of Game Levels
# The distribution of game levels shows that the majority of the events are happening in the middle levels of the game (around level 10). This could suggest that most users progress to these levels before stopping, or that these levels simply have more interactive events.

# In[18]:


# Plot the distribution of levels
plot_count(train_data, 'level', 'Distribution of Game Levels', 'green')


# #### Distribution of Level Groups
# The level group distribution shows that the majority of events belong to the '5-12' level group. This is consistent with the distribution of game levels, as the majority of events occurred at these levels.

# In[19]:


# Plot the distribution of level groups
plot_count(train_data, 'level_group', 'Distribution of Level Groups', 'red')


# ### Interesting Insights

# In[20]:


# Analyzing 'screen_coor_x' and 'screen_coor_y'
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(train_data['screen_coor_x'].dropna(), ax=ax[0], bins=30, kde=True, color='skyblue')
ax[0].set_title('Distribution of screen_coor_x')

sns.histplot(train_data['screen_coor_y'].dropna(), ax=ax[1], bins=30, kde=True, color='skyblue')
ax[1].set_title('Distribution of screen_coor_y')

plt.show()


# In[21]:


# Scatter plot for 'screen_coor_x' and 'screen_coor_y'
plt.figure(figsize=(6, 6))
plt.scatter(train_data['screen_coor_x'], train_data['screen_coor_y'], alpha=0.2)
plt.title('Scatter Plot of screen_coor_x vs screen_coor_y')
plt.xlabel('screen_coor_x')
plt.ylabel('screen_coor_y')
plt.show()


# In[22]:


def filter_data(df, column_name, values):
    """Filter rows where column_name is in values."""
    return df[df[column_name].isin(values)]

def calculate_statistics(df, column_name):
    """Calculate mean and median of a column."""
    mean_val = df[column_name].mean()
    median_val = df[column_name].median()
    return mean_val, median_val

def bin_column_data(df, column_name, bins):
    """Bin the column data into ranges and count the number of data points in each bin."""
    binned_data = pd.cut(df[column_name], bins=bins)
    binned_counts = binned_data.value_counts().sort_index()
    return binned_counts

def correct_xticklabels(bins):
    """Create labels for the x-axis."""
    return [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins)-1)]

def plot_binned_counts(binned_counts, bins, mean_val, median_val, title):
    """Plot the binned counts."""
    plt.figure(figsize=(20, 9))
    g = sns.barplot(x=binned_counts.index.astype(str), y=binned_counts.values, color='blue')
    plt.title(title, fontsize=18)
    g.set_xticklabels(correct_xticklabels(bins), rotation=90)
    g.set(xlabel='hover_duration, ms', ylabel='Count')
    g.axvline(x=np.digitize(mean_val, bins=bins)-1, color="red")
    g.text(np.digitize(mean_val, bins=bins)-1, 140000, f'Average ={round(mean_val, 1)}', rotation=90)
    g.axvline(x=np.digitize(median_val, bins=bins)-1, color="red")
    g.text(np.digitize(median_val, bins=bins)-1, 140000, f'Median ={round(median_val, 1)}', rotation=90)
    plt.show()

# Usage
hover_data = filter_data(train_data, 'event_name', ['object_hover', 'map_hover'])
mean_hover_duration_train, median_hover_duration_train = calculate_statistics(hover_data, 'hover_duration')
# Define the bin edges
bins = np.arange(0, 50001, 1000)
hover_duration_train_binned = bin_column_data(hover_data, 'hover_duration', bins)

plot_binned_counts(hover_duration_train_binned, bins, mean_hover_duration_train, median_hover_duration_train, 'hover_duration for events in train_dataset')


# In[23]:


# Analyzing 'elapsed_time'
plt.figure(figsize=(6, 6))
sns.histplot(train_data['elapsed_time'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of elapsed_time')
plt.xlabel('elapsed_time')
plt.show()


# #### Elapsed Time Statistics
# From the histogram of 'elapsed_time', we can observe that the distribution is heavily skewed to the right, with a few sessions having unusually high elapsed time values. These could potentially be outliers or errors in the data.

# In[24]:


# Display statistics related to elapsed time
elapsed_time_stats = train_data['elapsed_time'].describe()
elapsed_time_stats

# Plot the distribution of 'elapsed_time'
plt.figure(figsize=(10, 6))
sns.histplot(train_data['elapsed_time'], bins=100, color='purple')
plt.title('Distribution of Elapsed Time', size=15)
plt.xlabel('Elapsed Time (in milliseconds)', size=11)
plt.ylabel('Count', size=11)
plt.show()


# From the table below it shows subset of the data that falls in the top 1% of 'elapsed_time' which could suggest outliers.

# In[25]:


# Check the values on the high end of 'elapsed_time'
high_elapsed_time = train_data[train_data['elapsed_time'] > train_data['elapsed_time'].quantile(0.99)]
high_elapsed_time


# So I will set all 'elapsed_time' values above the 99th percentile to the 99th percentile value. This would limit the effect of extreme values without completely removing them from the dataset.

# In[26]:


# Cap 'elapsed_time' at the 99th percentile
train_data['elapsed_time'] = train_data['elapsed_time'].clip(upper=train_data['elapsed_time'].quantile(0.99))


# The maximum value is now significantly lower than before, while the other statistics (mean, standard deviation, etc.) remain similar. This means that the extreme high values have been limited, which should help to reduce their influence on the model.

# In[27]:


# Verify the change
train_data['elapsed_time'].describe()


# ### Multivariate analysis 
# Examine the interaction and relationship between the different features of the dataset

# In[28]:


# Calculate the correlation matrix
corr = train_data.corr()

# Plotting the heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True, 
        cmap='coolwarm') 
plt.title('Correlation Heatmap')
plt.show()


# ### Feature engineering functions

# In[29]:


# Reference: https://www.kaggle.com/code/gusthema/student-performance-w-tensorflow-decision-forests

CATEGORICAL = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
NUMERICAL = ['elapsed_time','level','page','room_coor_x', 'room_coor_y',
        'screen_coor_x', 'screen_coor_y', 'hover_duration']

def feature_engineer(dataset_df):
    dfs = []
    for c in CATEGORICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('mean')
        dfs.append(tmp)
    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)
    dataset_df = pd.concat(dfs,axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id')
    return dataset_df


# In[30]:


CATEGORICAL = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
NUMERICAL = ['elapsed_time','level','page','room_coor_x', 'room_coor_y',
        'screen_coor_x', 'screen_coor_y', 'hover_duration']
BINNING = ['elapsed_time', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']

# Define feature engineering function
def feature_engineer_ver2(dataset_df):
    dfs = []
    for c in CATEGORICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = c + '_nunique'
        dfs.append(tmp)

    for c in NUMERICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('mean')
        tmp.name = c + '_mean'
        dfs.append(tmp)

        # Compute standard deviation only for certain features
        if c in BINNING:
            tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('std')
            tmp.name = c + '_std'
            dfs.append(tmp)

        # Binning
        if c in BINNING:  # Check if column is in the list of columns to bin
            dataset_df[c+'_bin'] = pd.qcut(dataset_df[c], q=4, duplicates='drop')
            tmp = dataset_df.groupby(['session_id','level_group'])[c+'_bin'].agg('count')
            tmp.name = c + '_bin_count'
            dfs.append(tmp)

    # Interaction between screen coordinates
    if 'screen_coor_x' in NUMERICAL and 'screen_coor_y' in NUMERICAL:
        # Compute Euclidean distance instead of product
        dataset_df['screen_coor'] = np.sqrt(dataset_df['screen_coor_x']**2 + dataset_df['screen_coor_y']**2)
        tmp = dataset_df.groupby(['session_id','level_group'])['screen_coor'].agg(['mean', 'std'])
        tmp.columns = ['screen_coor_mean', 'screen_coor_std']
        dfs.append(tmp)

    # Aggregated features
    if 'hover_duration' in NUMERICAL:
        dataset_df['total_hover_duration'] = dataset_df.groupby(['session_id'])['hover_duration'].transform('sum')
        tmp = dataset_df.groupby(['session_id','level_group'])['total_hover_duration'].agg('mean')
        tmp.name = 'total_hover_duration_mean'
        dfs.append(tmp)

    dataset_df = pd.concat(dfs,axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id')

    dataset_df['screen_coor_mean'] = dataset_df['screen_coor_mean'].astype('int32')
    for col in dataset_df.select_dtypes(include='float16').columns:
        dataset_df[col] = dataset_df[col].astype('float32')

    return dataset_df


# In[31]:


CATEGORICAL = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
NUMERICAL = ['elapsed_time','level','page','room_coor_x', 'room_coor_y',
        'screen_coor_x', 'screen_coor_y', 'hover_duration']
BINNING = ['elapsed_time', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']

from sklearn.preprocessing import PowerTransformer

def feature_engineer_ver3(dataset_df):
    dfs = []
    pt = PowerTransformer(method='yeo-johnson')

    for c in CATEGORICAL:
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = c + '_nunique'
        dfs.append(tmp)

        # Create dummy variables for top N most common events and names
        top_N = dataset_df[c].value_counts()[:10].index
        for val in top_N:
            dataset_df[c + '_' + val] = (dataset_df[c] == val).astype(int)
        tmp = dataset_df.groupby(['session_id','level_group']).agg({c + '_' + val: 'sum' for val in top_N})
        dfs.append(tmp)

    for c in NUMERICAL:
        # Fill missing values with the column median
        dataset_df[c].fillna(dataset_df[c].median(), inplace=True)

        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('mean')
        tmp.name = c + '_mean'
        dfs.append(tmp)

        # Compute standard deviation only for certain features
        if c in BINNING:
            tmp = dataset_df.groupby(['session_id','level_group'])[c].agg('std')
            tmp.name = c + '_std'
            dfs.append(tmp)

        # Normalize 'elapsed_time' column
        if c == 'elapsed_time':
            dataset_df[c] = pt.fit_transform(dataset_df[[c]])

        # Binning
        if c in BINNING:  # Check if column is in the list of columns to bin
            dataset_df[c+'_bin'] = pd.qcut(dataset_df[c], q=4, duplicates='drop')
            #dataset_df[c+'_bin'] = pd.qcut(dataset_df[c], q=4, duplicates='drop').astype('category')

            tmp = dataset_df.groupby(['session_id','level_group'])[c+'_bin'].agg('count')
            tmp.name = c + '_bin_count'
            dfs.append(tmp)

    # Interaction between screen coordinates
    if 'screen_coor_x' in NUMERICAL and 'screen_coor_y' in NUMERICAL:
        # Compute Euclidean distance instead of product
        dataset_df['screen_coor'] = np.sqrt(dataset_df['screen_coor_x']**2 + dataset_df['screen_coor_y']**2)
        tmp = dataset_df.groupby(['session_id','level_group'])['screen_coor'].agg(['mean', 'std'])
        tmp.columns = ['screen_coor_mean', 'screen_coor_std']
        dfs.append(tmp)

    # Aggregated features
    if 'hover_duration' in NUMERICAL:
        dataset_df['total_hover_duration'] = dataset_df.groupby(['session_id'])['hover_duration'].transform('sum')
        tmp = dataset_df.groupby(['session_id','level_group'])['total_hover_duration'].agg('mean')
        tmp.name = 'total_hover_duration_mean'
        dfs.append(tmp)

    dataset_df = pd.concat(dfs, axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id')

    return dataset_df



# In[32]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

CATEGORICAL = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
NUMERICAL = ['elapsed_time','level','page','room_coor_x', 'room_coor_y',
             'screen_coor_x', 'screen_coor_y', 'hover_duration']

def feature_engineer_ver4(dataset_df):
    dfs = []
    le = LabelEncoder()
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

    for c in CATEGORICAL:
        # Label encoding for categorical features
        dataset_df[c+'_encoded'] = le.fit_transform(dataset_df[c].astype(str))
        tmp = dataset_df.groupby(['session_id','level_group'])[c+'_encoded'].agg(['mean', 'std'])
        tmp.columns = [c + '_encoded_mean', c + '_encoded_std']
        dfs.append(tmp)

    for c in NUMERICAL:
        # Fill missing values with the column median
        dataset_df[c].fillna(dataset_df[c].median(), inplace=True)

        # Calculate sum, mean and std for numerical features
        tmp = dataset_df.groupby(['session_id','level_group'])[c].agg(['sum', 'mean', 'std'])
        tmp.columns = [c + '_sum', c + '_mean', c + '_std']
        dfs.append(tmp)

        # Apply binning to numerical features
        dataset_df[c+'_binned'] = discretizer.fit_transform(dataset_df[[c]])
        tmp = dataset_df.groupby(['session_id','level_group'])[c+'_binned'].agg(['mean', 'std'])
        tmp.columns = [c + '_binned_mean', c + '_binned_std']
        dfs.append(tmp)

    # Interaction between screen coordinates
    if 'screen_coor_x' in NUMERICAL and 'screen_coor_y' in NUMERICAL:
        # Compute Euclidean distance instead of product
        dataset_df['screen_coor'] = np.sqrt(dataset_df['screen_coor_x']**2 + dataset_df['screen_coor_y']**2)
        tmp = dataset_df.groupby(['session_id','level_group'])['screen_coor'].agg(['sum', 'mean', 'std'])
        tmp.columns = ['screen_coor_sum', 'screen_coor_mean', 'screen_coor_std']
        dfs.append(tmp)

    # Aggregated features
    if 'hover_duration' in NUMERICAL:
        dataset_df['total_hover_duration'] = dataset_df.groupby(['session_id'])['hover_duration'].transform('sum')
        tmp = dataset_df.groupby(['session_id','level_group'])['total_hover_duration'].agg(['mean', 'std'])
        tmp.columns = ['total_hover_duration_mean', 'total_hover_duration_std']
        dfs.append(tmp)

    dataset_df = pd.concat(dfs, axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id')

    dataset_df['page_sum'] = dataset_df['page_sum'].astype('int32')
    for col in dataset_df.select_dtypes(include='float16').columns:
        dataset_df[col] = dataset_df[col].astype('float32')

    return dataset_df


# In[33]:


CATEGORICAL = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
NUMERICAL = ['elapsed_time','level','page','room_coor_x', 'room_coor_y',
             'screen_coor_x', 'screen_coor_y', 'hover_duration']

def feature_engineer_ver5(df):
    dfs = []
    le = LabelEncoder()
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

    for c in CATEGORICAL:
        # Label encoding for categorical features
        df[c+'_encoded'] = le.fit_transform(df[c].astype(str))
        tmp = df.groupby(df.index)[c+'_encoded'].agg(['mean', 'std'])
        tmp.columns = [c + '_encoded_mean', c + '_encoded_std']
        dfs.append(tmp)

    for c in NUMERICAL:
        # Fill missing values with the column median
        df[c].fillna(df[c].median(), inplace=True)

        # Calculate sum, mean and std for numerical features
        tmp = df.groupby(df.index)[c].agg(['sum', 'mean', 'std'])
        tmp.columns = [c + '_sum', c + '_mean', c + '_std']
        dfs.append(tmp)

        # Apply binning to numerical features
        df[c+'_binned'] = discretizer.fit_transform(df[[c]])
        tmp = df.groupby(df.index)[c+'_binned'].agg(['mean', 'std'])
        tmp.columns = [c + '_binned_mean', c + '_binned_std']
        dfs.append(tmp)

    # Interaction between screen coordinates
    if 'screen_coor_x' in NUMERICAL and 'screen_coor_y' in NUMERICAL:
        # Compute Euclidean distance instead of product
        df['screen_coor'] = np.sqrt(df['screen_coor_x']**2 + df['screen_coor_y']**2)
        tmp = df.groupby(df.index)['screen_coor'].agg(['sum', 'mean', 'std'])
        tmp.columns = ['screen_coor_sum', 'screen_coor_mean', 'screen_coor_std']
        dfs.append(tmp)

    # Aggregated features
    if 'hover_duration' in NUMERICAL:
        df['total_hover_duration'] = df.groupby(df.index)['hover_duration'].transform('sum')
        tmp = df.groupby(df.index)['total_hover_duration'].agg(['mean', 'std'])
        tmp.columns = ['total_hover_duration_mean', 'total_hover_duration_std']
        dfs.append(tmp)

    df = pd.concat(dfs, axis=1)
    df = df.fillna(-1)

    df['hover_duration_sum'] = df['hover_duration_sum'].astype('int32')
    for col in df.select_dtypes(include='float16').columns:
        df[col] = df[col].astype('float32')

    return df



# In[34]:


dataset_df = feature_engineer(train_data)


# In[ ]:


# Replace `inf` values:
dataset_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace `NaN` values with column mean:
dataset_df.fillna(dataset_df.mean(), inplace=True)


# ### Measurement of feature engineering functions

# In[ ]:


import tensorflow as tf
import tensorflow_decision_forests as tfdf


# In[ ]:


def split_dataset(dataset, test_ratio=0.20):
    USER_LIST = dataset.index.unique()
    split = int(len(USER_LIST) * (1 - 0.20))
    return dataset.loc[USER_LIST[:split]], dataset.loc[USER_LIST[split:]]

train_x, valid_x = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_x), len(valid_x)))


# In[ ]:


tfdf.keras.get_all_models()


# In[ ]:


VALID_USER_LIST = valid_x.index.unique()
prediction_df = pd.DataFrame(data=np.zeros((len(VALID_USER_LIST),18)), index=VALID_USER_LIST)
models = {}
# Create an empty dictionary to store the evaluation score for each question.
evaluation_dict ={}


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

models = {}  # To store the trained models
evaluation_dict = {}  # To store the evaluation accuracies
feature_importances = {}  # To store the feature importances

for q_no in range(1, 19):

    # Select level group for the question based on the q_no.
    if q_no<=3:
        grp = '0-4'
    elif q_no<=13:
        grp = '5-12'
    else:
        grp = '13-22'
    print("### q_no", q_no, "grp", grp)

    # Filter the rows in the datasets based on the selected level group.
    train_df = train_x.loc[train_x.level_group == grp]
    train_users = train_df.index.values
    valid_df = valid_x.loc[valid_x.level_group == grp]
    valid_users = valid_df.index.values

    # Select the labels for the related q_no.
    train_labels = labels.loc[labels.q==q_no].set_index('session').loc[train_users]
    valid_labels = labels.loc[labels.q==q_no].set_index('session').loc[valid_users]

    # Add the label to the filtered datasets.
    train_df["correct"] = train_labels["correct"]
    valid_df["correct"] = valid_labels["correct"]

    # Drop the 'level_group' feature
    train_df = train_df.drop(columns=['level_group'])
    valid_df = valid_df.drop(columns=['level_group'])

    # Split your data into features (X) and target (y)
    X_train = train_df.drop('correct', axis=1)
    y_train = train_df['correct']
    X_valid = valid_df.drop('correct', axis=1)
    y_valid = valid_df['correct']

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Store the model
    models[f'{grp}_{q_no}'] = model

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for visualization
    feature_importances[q_no] = pd.DataFrame({"Feature": X_train.columns, "Importance": importances}).sort_values("Importance", ascending=False)

    # Evaluate the model
    y_pred = model.predict(X_valid)
    evaluation_dict[q_no] = accuracy_score(y_valid, y_pred)

# Display the feature importances and evaluations
for q_no in range(1, 19):
    print(f"Question {q_no}")
    print(feature_importances[q_no])
    print(f"Evaluation accuracy: {evaluation_dict[q_no]}")

