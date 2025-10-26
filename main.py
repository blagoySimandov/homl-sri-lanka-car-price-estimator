# %% [markdown]
# # Load the dataset
# %%
import pandas as pd
import os

# Load the dataset
df = pd.read_csv('dataset_vehicles.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
df.head()

# %%
df.info()

# %%
df[df.duplicated()]
#they posted their car twice. Nuke them :D

# %%
df.drop_duplicates().head()



# %% [markdown]
# hmm but this will drop  the duplicates that are duplicates for all columns. what if somebday marked his car with one column difference:?
#

# %%
df[df.duplicated(subset=['Title', "Sub_title","Price","Description"], keep=False)].sort_values(by=['Title', 'Sub_title',"Description"])


# %%
df["Post_URL"]

# %% [markdown]
# ### Post url seems useless might drop it

# %%
df["Seller_name"].value_counts()

# %% [markdown]
# ### Seller name could be useful, maybe there is a correlation between names of big sellers and and car prices ?
# Would be worth it to Pick the biggest N sellers and make everybody else have an "Unknown Seller"  type ?... Hmmm

# %% [markdown]
# The lengths is 18938 and we cans ee that there are null fields in the fields: Edition (13908/18938 entries) and Body (17038/18938). Due to Edition having too many nulls we can't really drop them... So we might have to think of another way to handle this. Let's look at the value counts of Edition.

# %%
#how many sellers are seen more than 10 times
df[df["Seller_name"].isin(df["Seller_name"].value_counts()[df["Seller_name"].value_counts() > 3].index)]


# %%
df["Edition"].value_counts()

# %% [markdown]
# Hmm seems like a free text column. Might be worth normalizing it a bit. Maybe at least make everything lower case and run a bag of wrods on it ?

# %%
edition_lowercase = df["Edition"].str.lower()

# %%
edition_lowercase.value_counts()

# %% [markdown]
# Okay... this helped we got the number down by almost 1000... But we could do better...
# I will leave this for now..
#

# %%
df[df["Edition"].isin(df["Edition"].value_counts()[df["Edition"].value_counts() > 1].index)]

# %% [markdown]
# Okay this is getting nowhere..
# We might try and to do a text vocabulary on the Edition
# similar to what tf transformers do:
#
# https://www.tensorflow.org/tfx/tutorials/transform/census

# %%
df["Body"].value_counts()

# %% [markdown]
# Seems clean enough 

# %%
df["Fuel"].value_counts()

# %% [markdown]
# Clean as well.... Although not sure what Other Fuel type is

# %%
df[df["Fuel"]=="Other fuel type"].head()

# %%
df["Capacity"].value_counts()

# %%
#after trying to clean this i got a problem where magically everything became Null :D
#so im now going back to see whetehr evertyhign follows teh format <NUMBER cc> :DDDDD

# %%
pattern = r'^\d+(\.\d+)?\s+cc$'
non_matching = df[~df["Capacity"].str.match(pattern, na=False)]
non_matching[["Capacity"]]


# %% [markdown]
# Okay yeah it's the commas that are the problem... I will just nuke them in the data cleaning layer

# %%
df.info()

# %% [markdown]
# We can see that Body has null columns and Edition.... The null columns of body could be dropped as theya aren't many
# but the ones for edition definietly shouldn't as they are too many...
# Now looking at edition it does feel weird that it is full of rubbish...
# I might try cleaning this column a bit but since it is pretty much freetext it would be hard to impute it in a sensible manenr.
#
# I google away and saw this comment on a reddit post: https://www.reddit.com/r/learnprogramming/comments/1m75jgd/how_should_i_handle_missing_data_in_both/
# ```
# If your text is free-form paragraphs, that is more complicated. There you can replace the missing word with a token that preserves the data point and allows the model to learn that the token signifies missing info. In essence, for free-form text you can use a language model to predict the most likely missing word or phrase by considering the surrounding text.
#
# One bit of advice is that sometimes the best way is to simply try different imputation methods on the train set and test it on the test set and compare results. Then choose the best one.
# ```
# So I might just leave it for now and try different ways to impute it later...
# Although i will test it on the validation set since we don't really want any  leakadge... :D
#
#
#
# Also  a lot of actually numerical columns are strings.
# I should clean them up and turn them to numbers:
# Price, Mileage, Capacity, Date (unix ?)
#
# I should conider the location as well. I might be able to turn it into lat lon ? 
# I will look into this alter after i have  a cleaner dataset and i have experimented a bit...
#

# %%

df["Brand"].value_counts()

# %% [markdown]
# ## Data cleaning

# %% [markdown]
# Things to do:
#
# [ ] Convert Capacity to number by creating a transformer that removes " cc" at the end
#
# [ ] Convert Mileage to number by creating a transformer that removes " km" at the end
#
# [ ] Convert Price to number by moving it removing the preprended "Rs " from it
#
# [ ] Make all text field lowercase string, remove urls
#
# [ ] Remove any duplicates
#
# [ ] Drop Subtitle as it is just a free text version of concated posted date + location
#
# [ ] Consider dropping Title as it is just Brand + Model + Edition + Year + "for sale"
#

# %% [markdown]
# The biggest challenge would probably be Handling the categorical values that are all over the place.

# %% [markdown]
# ## Data Cleaning Pipeline

# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import re

#implementation with transformerMixin inspired by: https://stackoverflow.com/questions/65488758/scikit-learn-pipeline-custom-transformer-function
class NumericCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_config):
        self.columns_config = columns_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, pattern in self.columns_config.items():
            if col in X.columns:
                X[col] = X[col].astype(str).str.replace(pattern, '', regex=True).str.strip()
                X[col] = X[col].str.replace(',', '', regex=False)
                X[col] = pd.to_numeric(X[col], errors='coerce')
        return X
        
#removes urls + makes strings to lowercase
class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        #copy pasted this from here: https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
        url_pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'

        for col in self.text_columns:
            if col in X.columns:
                X[col] = X[col].astype(str).str.lower()
                X[col] = X[col].str.replace(url_pattern, '', regex=True).str.strip()
        return X
        
#removes duplicates when subset_columns match, ignores the other columns if subset_columns is present 
class DuplicateRemoverTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        duplicates = X.duplicated(keep=False)
        X = X.drop(X[duplicates].index)
        return X.reset_index(drop=True)
        
#nukes columns
class ColumnNukerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        existing_cols = [col for col in self.columns_to_drop if col in X.columns]
        return X.drop(columns=existing_cols)

class ColumnRenamerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rename_mapping):
        self.rename_mapping = rename_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        existing_mappings = {old: new for old, new in self.rename_mapping.items() if old in X.columns}
        return X.rename(columns=existing_mappings)

# %%
cleaning_pipeline = Pipeline([
    ('drop_columns', ColumnNukerTransformer(
        columns_to_drop=['Post_URL', 'Title', 'Sub_title']
    )),
    ('remove_duplicates', DuplicateRemoverTransformer()),
    ('clean_numeric', NumericCleanerTransformer(
        columns_config={
            'Capacity': r'\s*cc\s*$',
            'Mileage': r'\s*km\s*$',
            'Price': r'^Rs\s*'
        }
    )),
    ('clean_text', TextCleanerTransformer(
        text_columns=['Edition', 'Description', 'Seller_name',"Location","Seller_type"]
    )),
    ('rename_columns', ColumnRenamerTransformer(
        rename_mapping={'published_date': 'Published_Date'}
    )),
])

# %%
df_cleaned = cleaning_pipeline.fit_transform(df)

# %%
df_cleaned.head()

# %%
df_cleaned.info()

# %% [markdown]
#
