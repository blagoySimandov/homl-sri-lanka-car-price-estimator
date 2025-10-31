# %% [markdown]
# # Load the dataset and intial "exploration"
# %%
import pandas as pd

df = pd.read_csv("dataset_vehicles.csv")

df.head()

# %%
df.info()

# %%
df[df.duplicated()]
# they posted their car twice. Nuke them :D

# %%
# df.drop_duplicates().head()


# %% [markdown]
# Hmm but this will drop  the duplicates that are duplicates for all columns. What if somebday marked his car with one column difference ?
# Example: Same description (probably the same car)? Or even better Same Description + Title + Subtitle (almost definetly same car)
#

# %%
df[df.duplicated(subset=["Description"], keep=False) & ~df.duplicated()].sort_values(by=["Description"]).head()


# %%
df["Post_URL"]

# %% [markdown]
# Post url seems useless might drop it

# %%
df["Seller_name"].value_counts()

# %% [markdown]
# -----------
# Seller name could be useful, maybe there is a correlation between names of big sellers and and car prices ?
# Would be worth it to Pick the biggest N sellers and make everybody else have an "Unknown Seller"  type ?

# %%
df.info()

# %% [markdown]
# The lengths is 18938 and we can see that there are null fields in the fields: Edition (13908/18938 entries) and Body (17038/18938). Due to Edition having too many nulls we can't really drop them... So we might have to think of another way to handle this. Let's look at the value counts of Edition.

# %%
df["Edition"].value_counts()

# %% [markdown]
# Hmm seems like a free text column. Might be worth normalizing it a bit. Maybe at least make everything lower case and run a bag of wrods on it ?

# %%
# how many sellers are seen more than 10 times
df[df["Seller_name"].isin(df["Seller_name"].value_counts()[df["Seller_name"].value_counts() > 3].index)]


# %% [markdown]
#  12198 sellers out of 18k are seen more than 3 times. Not bad. Maybe differentiating the dealerships from the normal sellers might be worth it? We can either check this in the EDA stage by seeing if there is a price correlation between the two (size of dealership to price) or just put it in the model and have feature selection take care of it. I will decide on the approach later.

# %% [markdown]
# Checking if the description contains pricing information. We could technically use this. but I feel like it's (only 1000 rows have it and we aren't even sure whether it's accurate and it is talking about the price)

# %%
df[df["Description"].str.contains("Rs ", na=True)]

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
# Clean as well.... Although not sure what "Other Fuel type" is

# %%
df[df["Fuel"] == "Other fuel type"].head()

# %%
df["Capacity"].value_counts()

# %%
# after trying to clean this i got a problem where magically everything became Null :D
# so im now going back to see whetehr evertyhing follows the format <NUMBER cc> :D

# %%
pattern = r"^\d+(\.\d+)?\s+cc$"
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
# Although i will test it on the validation set since we don't really want any  leakadge...
#
#
#
# Also  a lot of actually numerical columns are strings.
# I should clean them up and turn them to numbers:
# Price, Mileage, Capacity, Date (unix ?)
#
# I should conider the location as well. I might be able to turn it into lat lon ?
# I will look into this alter I have  a cleaner dataset and i have experimented a bit...
#

# %%

df["Brand"].value_counts()

# %%
len(df[df.apply(lambda row: str(row["Price"]) in str(row["Description"]), axis=1)])  # check if the price is in a lot of descriptions

# %% [markdown]
# ## Data cleaning

# %% [markdown]
# Things to do:
#
# [ ] Convert Capacity to number by creating a transformer that removes " cc" at the end
#
# [ ] Nuke Seller_type because everyone is a premium member (i found this out in EDA)
#
# [ ] Convert Mileage to number by creating a transformer that removes " km" at the end
#
# [ ] Convert Price to number by moving it removing the preprended "Rs " from it
#
# [ ] Make all text field lowercase string, remove urls
#
# [ ] Remove any duplicates (maybe consider removing duplicates using a subset columns array)
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


# implementation with transformerMixin inspired by: https://stackoverflow.com/questions/65488758/scikit-learn-pipeline-custom-transformer-function
class NumericCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_config):
        self.columns_config = columns_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, pattern in self.columns_config.items():
            if col in X.columns:
                X[col] = X[col].astype(str).str.replace(pattern, "", regex=True).str.strip()
                X[col] = X[col].str.replace(",", "", regex=False)
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X


# removes urls + makes strings to lowercase
class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # copy pasted this from here: https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
        url_pattern = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"

        for col in self.text_columns:
            if col in X.columns:
                X[col] = X[col].apply(lambda v: v.lower() if isinstance(v, str) else v)
                X[col] = X[col].replace(url_pattern, "", regex=True).str.strip()

        return X


class DuplicateRemoverTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        duplicates = X.duplicated(keep=False)
        X = X.drop(X[duplicates].index)
        return X.reset_index(drop=True)


# nukes columns
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


class DatetimeToUnixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_columns):
        self.datetime_columns = datetime_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.datetime_columns:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col]).astype(int) / 10**9
        return X


# %%
cleaning_pipeline = Pipeline(
    [
        ("drop_columns", ColumnNukerTransformer(columns_to_drop=["Post_URL", "Title", "Sub_title", "Seller_type"])),
        ("remove_duplicates", DuplicateRemoverTransformer()),
        (
            "clean_numeric",
            NumericCleanerTransformer(
                columns_config={
                    "Capacity": r"\s*cc\s*$",
                    "Mileage": r"\s*km\s*$",
                    "Price": r"^Rs\s*",
                }
            ),
        ),
        ("clean_text", TextCleanerTransformer(text_columns=["Edition", "Description", "Seller_name"])),
        (
            "rename_columns",
            ColumnRenamerTransformer(
                rename_mapping={
                    "published_date": "Published_Date",  # keep consistent naming pattern
                }
            ),
        ),
        ("convert_datetime", DatetimeToUnixTransformer(datetime_columns=["Published_Date"])),
    ]
)


# %%
df_cleaned = cleaning_pipeline.fit_transform(df)

# %%
df_cleaned.head()

# %%
df_cleaned.info()

# %%
df_cleaned[df_cleaned["Edition"].isna()].head()


# %% [markdown]
# --------------
# I should look at this in EDA and see if there is a correlation between Edition missing ?
# The question is:
#
# MCAR, MAR or MNAR: (taken from this https://medium.com/%40ajayverma23/data-imputation-a-comprehensive-guide-to-handling-missing-values-b5c7d11c3488)
#
# MCAR = Missing Completely At Random (the missingness has nothing to do with values or other features)
#
# MAR = Missing At Random (missingness depends on other observed variables)
#
# MNAR = Missing Not At Random

# %% [markdown]
# # Train-Test Split

# %%
from sklearn.model_selection import train_test_split

X = df_cleaned.drop("Price", axis=1)
y = df_cleaned["Price"]
rngs = 42  # using this so we have reproduceability

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rngs)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=rngs
)  # 25 percent so that validation and test set are the same size

df_train = X_train.copy()
df_train["Price"] = y_train

df_val = X_val.copy()
df_val["Price"] = y_val

df_test = X_test.copy()
df_test["Price"] = y_test

print(f"training set size: {len(df_train)}")
print(f"validation set size: {len(df_val)}")
print(f"test set size: {len(df_test)}")

# %% [markdown]
# # EDA - Exploritary Data Analysis

# %% [markdown]
# There are a few things I want to check out.
#
# 1. Check for skewed values
#
# 2. Check for relationships between columns
#     Price to Body,Model,Year, Capacity, Mileage, Fuel Type
# 3. Try some new features like: Car age and Mileage per year and see how predictive they are of the price. (check for non linear relationships as well)
#

# %%
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# Using a copy to avoid polluting the training dataframe during EDA experiments

# %%
df_train_copy = df_train.copy()

# %%
df_train_copy.describe()

# %%
from scipy.stats import skew

# https://www.geeksforgeeks.org/python/scipy-stats-skew-python/
# Highly skeweed values (over 1) need to be transformed.
skew(df_cleaned["Mileage"]), skew(df_cleaned["Price"])

# %% [markdown]
# --------------------------------------
# Let's take a look at some outliers.

# %%
cols = ["Mileage", "Capacity", "Price"]

plt.figure(figsize=(10, 8))

for i, col in enumerate(cols, 1):
    plt.subplot(3, 1, i)
    plt.scatter(df_train_copy.index, df_train_copy[col], alpha=0.6)
    plt.title(col)
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.grid(True)

plt.tight_layout()
plt.show()


# %%
df_train_copy.sort_values("Price",ascending=False).head()

# %%
df_train_copy.sort_values("Mileage",ascending=False).head()

# %%
df_train_copy.sort_values("Capacity",ascending=False).head()

# %% [markdown]
#
# There are a few outliers I need to clip. Seems like the point of clipping woiuld be around 1.0*1e8 or somewhere
# between 0.75 and 1 for price, 6000 for capacity and for mileage maybe between 0.6 and 0.8 * 0.e6
#
# ------------------
#
# I think I might actually do this in the data cleaning stage since some of the values (especially for price) just feel invalid.
# A used toyota  def isn't worth 179000000 (587422.15 USD)
#
# but some others  like the Range rovesr are accurate.
# For the capacity some are actually adequte wile others are just mistypings:
# Perodia Viva Elite is 1000 Capacity not 10 000 (one more zero)
#
# I think the best way to handle this is 1. Clip Capacityand Price in data cleaning.
#
# Capacity clipping point: 6500cc
# Price clipping point: 126500000 Rs (415k)
# Mileage: no clipping ( I don't think i can draw a good line between the outliers and acutal data..)
#
#
#

# %%
top_brands = df_train_copy["Brand"].value_counts().head(10).index
df_brand_price = df_train_copy[df_train_copy["Brand"].isin(top_brands)]

brand_data = [df_brand_price[df_brand_price["Brand"] == brand]["Price"] for brand in top_brands]

fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(brand_data, tick_labels=top_brands, vert=False)
ax.set_xlabel("Price")
ax.set_title("Price Distribution by Top 10 Brands")
plt.tight_layout()
plt.show()

# %% [markdown]
# Seems like a few brands are pretty representative (given the brand you can get a  a good price) this would be suzuki, nissan and honda.
# Others like land rover don't really give as a lot of data on the price.
#

# %%
top_models = df_train_copy["Model"].value_counts().head(15).index
df_model_price = df_train_copy[df_train_copy["Model"].isin(top_models)]

model_data = [df_model_price[df_model_price["Model"] == model]["Price"] for model in top_models]

fig, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(model_data, tick_labels=top_models, vert=False)
ax.set_xlabel("Price")
ax.set_title("Price Distribution by Top 15 Models")
plt.tight_layout()
plt.show()

# %% [markdown]
# Some do seem kinda representative but nothing really strikes me here.

# %% [markdown]
# ## Car Age to Price

# %%
current_year = 2025
df_train_copy["Car_Age"] = current_year - df_train_copy["Year"]

plt.figure(figsize=(14, 6))
plt.scatter(df_train_copy["Car_Age"], df_train_copy["Price"], alpha=0.3)
plt.xlabel("Car Age (years)")
plt.ylabel("Price (Rs)")
plt.title("Price vs Car Age")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Seems like a logarithmic relationship. With a really heavy tail. Tranforming it with a power tranformation will help for the parametric models like linear regression and ridge.

# %%

from scipy.stats import yeojohnson

df_train_copy["Car_Age_transformed"] = yeojohnson(df_train_copy["Car_Age"])[0]

# %%

correlation = df_train_copy["Car_Age_transformed"].corr(df_train_copy["Price"])
plt.scatter(df_train_copy["Car_Age_transformed"], df_train_copy["Price"], alpha=0.3)
plt.show()
print(correlation)

# %% [markdown]
# Better. Not the best

# %% [markdown]
# Transforming the target can sometimes be really beneficial  especially for parametric models:
# https://www.geeksforgeeks.org/machine-learning/powertransformer-in-scikit-learn/
#
# We just need to be careful and transform it back when we get the result from the model at the end
#

# %%
df_train_copy["Price_transformed"], lambda_price = yeojohnson(df_train_copy["Price"])


# %%

correlation = df_train_copy["Car_Age_transformed"].corr(df_train_copy["Price_transformed"])
plt.scatter(df_train_copy["Car_Age_transformed"], df_train_copy["Price_transformed"], alpha=0.3)
plt.show()
print(correlation)

# %% [markdown]
# Hmm this seems pretty good. Will try adding it in the grid search later

# %%
correlation_features = [
    "Price",
    "Year",
    "Mileage",
    "Capacity",
]
corr = df_train_copy[correlation_features].corr()
corr.style.background_gradient(cmap="coolwarm")

# %% [markdown]
# ----------------
# All of the numeric columns by themeselves have low correlation with the price. Some feature engineering or
# power transformations could help this. Will do some more analysis below and see.

# %% [markdown]
# ---------------
# Now I will check the distribution of all the numeric columns to see which would really need transformation.
#

# %%
correlation_features = ["Price", "Year", "Mileage", "Capacity"]

plt.figure(figsize=(10, 8))

for i, col in enumerate(correlation_features, 1):
    plt.subplot(2, 2, i)
    plt.hist(df_train_copy[col], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)

plt.tight_layout()
plt.show()


# %% [markdown]
# Okay this is pretty skewed.... A power transformer for all fo them might fix the issue.

# %%
df_train_copy["Mileage_transformed"] = yeojohnson(df_train_copy["Mileage"])[0]
df_train_copy["Capacity_transformed"] = yeojohnson(df_train_copy["Capacity"])[0]
df_train_copy["Year_transformed"] = yeojohnson(df_train_copy["Year"])[0]

# %%
corr = df_train_copy[["Price_transformed", "Car_Age_transformed", "Mileage_transformed", "Capacity_transformed", "Year_transformed"]].corr()
corr.style.background_gradient(cmap="coolwarm")

# %% [markdown]
# Correlations improve moderately after power transformations

# %%
import matplotlib.pyplot as plt

correlation_features = [
    "Price_transformed",
    "Car_Age_transformed",
    "Mileage_transformed",
    "Capacity_transformed",
    "Year_transformed",
]

plt.figure(figsize=(8, 15))  # taller figure for 5 stacked plots

colors = ["steelblue", "seagreen", "indianred", "darkorange", "mediumpurple"]

for i, (col, color) in enumerate(zip(correlation_features, colors), 1):
    plt.subplot(len(correlation_features), 1, i)
    plt.hist(df_train_copy[col], bins=30, color=color, edgecolor="black", alpha=0.7)
    plt.title(f"{col} Distribution", fontsize=12, fontweight="bold")
    plt.xlabel(col, fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# %% [markdown]
# ----------
# The distribution seems much more normal.
# Since parametric models like ridge and linear regression like this while knn and decision trees don't mind it. I will just leave it as is in the preprocessing step. Although i did do some experiments below in the grid search where I checked how much better transformed features perform compared to the untransformed once and the findings were staggering! The improvement was around 40-50% with the transformed  alues. From error around 2mil R.s to error around 1.1-1.2 mil R.s
#
# The only thing that doesn't really show a super normal distribution is Year but it has such good correlation with price now that i will leave it as is
#
# Note: it went from 0.35 to 0.67

# %% [markdown]
# ### Is Edition missing because of  MNAR,MCAR or MNAR ?

# %% [markdown]
# Approach inspired by: https://www.reddit.com/r/AskStatistics/comments/17nigqk/diagnosing_type_of_missing_data_mcar_mar_mnar/
#
# "What you're ultimately looking for is patterns in the missingness. In other words, do the records where a given feature is missing have a different distribution than the records where the feature isn't missing? If the answer to that question is a clear 'yes', then that feature isn't missing at random. A clear 'no' means the feature is missing completely random. If it's less clear then the feature then it can likely said to be missing at random, but could also go either way."

# %%
df_train_copy["Edition_missing"] = df_train_copy["Edition"].isna()

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

top_brands = df_train_copy["Brand"].value_counts().head(10).index
brand_missing = df_train_copy[df_train_copy["Brand"].isin(top_brands)].groupby("Brand")["Edition_missing"].mean().sort_values()
axes[0, 0].barh(brand_missing.index, brand_missing.values, color="steelblue")

axes[0, 1].bar(
    df_train_copy.groupby("Condition")["Edition_missing"].mean().index,
    df_train_copy.groupby("Condition")["Edition_missing"].mean().values,
    color="seagreen",
)

axes[1, 0].boxplot(
    [df_train_copy.loc[~df_train_copy["Edition_missing"], "Price"], df_train_copy.loc[df_train_copy["Edition_missing"], "Price"]],
    tick_labels=["Has Edition", "Missing"],
    vert=False,
)

axes[1, 1].boxplot(
    [df_train_copy.loc[~df_train_copy["Edition_missing"], "Year"], df_train_copy.loc[df_train_copy["Edition_missing"], "Year"]],
    tick_labels=["Has Edition", "Missing"],
    vert=False,
)

titles = ["Edition Missing by Brand", "Edition Missing by Condition", "Price Distribution", "Year Distribution"]
for ax, title in zip(axes.flat, titles):
    ax.set_title(title)

plt.tight_layout()
plt.show()


# %% [markdown]
# ---------
# Seems to be MAR or MCAR - Missing at Random / Missing at Completely Random
# I see a correlation between brand and  condition and their missing rate but it seems not super significant
# and also I'm not sure how to use that for the imputation...
# I will just impute it with a missing token "Unknown" or NaN or something similar
# I might be able to impute it by just getting the most common edition

# %% [markdown]
# ## Rare Category Analysis

# %%
high_cardinality_cols = ["Brand", "Model", "Location", "Seller_name"]

for col in high_cardinality_cols:
    print(f"\n{col} - Total unique values: {df_train[col].nunique()}")
    value_counts = df_train[col].value_counts()
    rare_threshold = len(df_train) * 0.002
    rare_categories = value_counts[value_counts < rare_threshold]
    print(f"Rare categories (< 0.2% of data or < {rare_threshold:.0f} occurrences): {len(rare_categories)}")
    print(f"Top 10 values:\n{value_counts.head(10)}")

# %% [markdown]
# -----------------------------
# Okay this is interesting. We could change all 'Rare' Brands to 'Rare' and encode them with one hot since there aren't that many unique...
#
# For model this won't work as there are too many. I think target encoding would be fine for this as it will give us the mean for the specific model.
#
# But for Seller_name this is a bit weird... I think a frequency encoding could work since this will allow us to deferentiate how "Big" a certain seller is.
#
# Which in my opinion is the most valuable information.
#
#
# Since there are too many of them.
#
# For the location  a target encoding would work wehll. As it *should* tell us the mean price fore a certain location.
#
#
# ----------------------------------

# %%
df_cleaned["Body"].value_counts()

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## Preprocessing Pipeline

# %% [markdown]
# -----------
# We need to:
#
# [X] Add the new transformed fields and drop the un transformed ones
#
# [X] Scale the  data (try different scalers in the grid search)
#
# [X] One hot encode the Fuel type
#
# [X] Ordinal Encode Condition field, Used, Reconditioned, New
#
# [X] One-hot encode Transmission field: Automatic, Manual, Tiptonic, Other transmission
#
# [X] Impute "Body" field with constant "Unknown" or something similiar
#
# [X] One-hot encode Body field: Hatchback,SUV / 4x4, Station wagon, MPV, CoupÃ©/Sports,Convertible
#
# [X] Bag of words the Description: https://stackoverflow.com/questions/30653642/combining-bag-of-words-and-other-features-in-one-model-using-sklearn-and-pandas
#
# [X] Impute Edition with unknown
#
# [X] Bag of words the Edition: Edition (could try and use a small vocab for this one, i think it would be worth it) or shove it into grid search ?
#

# %%
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    OrdinalEncoder,
    OneHotEncoder,
    TargetEncoder,
    PolynomialFeatures,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime


# %%
# adds carAge
class CarAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=2025):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Car_Age"] = self.current_year - X["Year"]
        return X.drop(columns=["Year"])


# %%
# not available in scikit learn so we need to create this one ourselves... :D
class FrequencyEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.frequency_mappings_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                freq_map = X[col].value_counts().to_dict()
                self.frequency_mappings_[col] = freq_map
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns and col in self.frequency_mappings_:
                X[col] = X[col].map(self.frequency_mappings_[col]).fillna(0)
        return X


# %%
preprocessing_pipeline_A = Pipeline(
    [
        ("add_car_age", CarAgeTransformer(current_year=2025)),
        (
            "target_encode",
            ColumnTransformer(
                [
                    (
                        "model_location_encoder",
                        TargetEncoder(categories="auto", target_type="continuous", smooth="auto", cv=5),
                        ["Model", "Location"],
                    )
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        ("frequency_encode", FrequencyEncoderTransformer(columns=["Seller_name"])),
        (
            "impute_missing",
            ColumnTransformer(
                [
                    (
                        "body_edition_imputer",
                        SimpleImputer(strategy="constant", fill_value="unknown"),
                        ["Body", "Edition"],
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "ordinal_encoding",
            ColumnTransformer(
                [
                    (
                        "condition_encoder",
                        OrdinalEncoder(
                            categories=[["used", "reconditioned", "new"]],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                        ["Condition"],
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "power_transform",
            ColumnTransformer(
                [
                    ("car_age_yj", PowerTransformer(method="yeo-johnson"), ["Car_Age"]),
                    ("mileage_yj", PowerTransformer(method="yeo-johnson"), ["Mileage"]),
                    (
                        "capacity_yj",
                        PowerTransformer(method="yeo-johnson"),
                        ["Capacity"],
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "one_hot_encoding",
            ColumnTransformer(
                [
                    (
                        "brand_encoder",
                        OneHotEncoder(
                            drop="first",
                            sparse_output=False,
                            handle_unknown="infrequent_if_exist",
                            min_frequency=0.005,
                        ),
                        ["Brand"],
                    ),
                    (
                        "fuel_encoder",
                        OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                        ["Fuel"],
                    ),
                    (
                        "transmission_encoder",
                        OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                        ["Transmission"],
                    ),
                    (
                        "body_encoder",
                        OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                        ["Body"],
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "text_features",
            ColumnTransformer(
                [
                    (
                        "description_bow",
                        CountVectorizer(max_features=50, lowercase=True, stop_words="english"),
                        "Description",
                    ),
                    (
                        "edition_bow",
                        CountVectorizer(max_features=30, lowercase=True, stop_words="english"),
                        "Edition",
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        ),
        ("feature_selection", None),
        ("scaler", None),
    ]
)

# %%
preprocessing_pipeline_B = Pipeline(
    [
        ("add_car_age", CarAgeTransformer(current_year=2025)),
        (
            "target_encode",
            ColumnTransformer(
                [
                    (
                        "model_location_encoder",
                        TargetEncoder(categories="auto", target_type="continuous", smooth="auto", cv=5),
                        ["Model", "Location"],
                    )
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        ("frequency_encode", FrequencyEncoderTransformer(columns=["Seller_name"])),
        (
            "impute_missing",
            ColumnTransformer(
                [
                    (
                        "body_edition_imputer",
                        SimpleImputer(strategy="constant", fill_value="unknown"),
                        ["Body", "Edition"],
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "ordinal_encoding",
            ColumnTransformer(
                [
                    (
                        "condition_encoder",
                        OrdinalEncoder(
                            categories=[["used", "reconditioned", "new"]],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                        ["Condition"],
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "poly_features",
            ColumnTransformer(
                [
                    (
                        "poly",
                        PolynomialFeatures(),
                        ["Mileage", "Capacity"],
                    )
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "power_transform",
            ColumnTransformer(
                [
                    ("car_age_yj", PowerTransformer(method="yeo-johnson"), ["Car_Age"]),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "one_hot_encoding",
            ColumnTransformer(
                [
                    (
                        "brand_encoder",
                        OneHotEncoder(
                            drop="first",
                            sparse_output=False,
                            handle_unknown="infrequent_if_exist",
                            min_frequency=0.005,
                        ),
                        ["Brand"],
                    ),
                    (
                        "fuel_encoder",
                        OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                        ["Fuel"],
                    ),
                    (
                        "transmission_encoder",
                        OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                        ["Transmission"],
                    ),
                    (
                        "body_encoder",
                        OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                        ["Body"],
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "text_features",
            ColumnTransformer(
                [
                    (
                        "description_bow",
                        CountVectorizer(max_features=50, lowercase=True, stop_words="english"),
                        "Description",
                    ),
                    (
                        "edition_bow",
                        CountVectorizer(max_features=30, lowercase=True, stop_words="english"),
                        "Edition",
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        ),
        ("feature_selection", None),
        ("scaler", None),
    ]
)

# %%
# since we need to transform the price now
#  y_pred = price_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()
# is needed  after pred.

# %% [markdown]
# # Model Selection
#
# After cleaning and preprocessing the data, we now need to find the best model for predicting car prices.
#
# All models will be tested with feature selection to find the optimal combination.

# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
import warnings

warnings.filterwarnings("ignore")

# %%
price_transformer = PowerTransformer(method="yeo-johnson")
y_train_transformed = price_transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_transformed = price_transformer.transform(y_val.values.reshape(-1, 1)).ravel()


# %%
def check_fit(model, X_train, y_train, X_val, y_val, price_transformer):
    y_train_pred_transformed = model.predict(X_train)
    y_train_pred = price_transformer.inverse_transform(y_train_pred_transformed.reshape(-1, 1)).ravel()

    y_val_pred_transformed = model.predict(X_val)
    y_val_pred = price_transformer.inverse_transform(y_val_pred_transformed.reshape(-1, 1)).ravel()

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)

    return train_mae, val_mae


# %% [markdown]
# ## Ridge Regression Experiments
#
# We start with Ridge regression because:
# 1. It handles multicollinearity well
# 2. Built-in regularization prevents overfitting
# 3. Works well with power-transformed features
#
# ### Feature Selection Strategy
#
# After the initial experiments showed ~1.3M MAE, we added feature selection to:
# - Reduce noise from irrelevant features
# - Combat overfitting (fewer features = simpler model)
# - Improve model generalization
#
# We test two approaches:
# - **TruncatedSVD**: Dimensionality reduction that captures latent patterns (great for text features)
#
# ### Variant A: No Polynomial Features
#
# This is our clean baseline. We apply:
# - Power transform to Car_Age, Mileage, Capacity
# - Target encoding for Model/Location
# - One-hot encoding for categorical features
# - Bag-of-words for text features
# - Optional feature selection (None, TruncatedSVD, or SelectKBest)

# %%
full_pipeline_A = Pipeline(
    [
        ("preprocessing", preprocessing_pipeline_A),
        ("model", Ridge()),
    ]
)

param_grid_A = {
    "preprocessing__scaler": [StandardScaler(), RobustScaler(), None],
    "preprocessing__text_features__description_bow__max_features": [50, 100],
    "preprocessing__text_features__edition_bow__max_features": [30, 50],
    "preprocessing__feature_selection": [
        None,
        TruncatedSVD(n_components=50),
        TruncatedSVD(n_components=100),
        SelectKBest(f_regression, k=50),
        SelectKBest(f_regression, k=100),
    ],
    "model__alpha": [0.01, 0.1, 1, 10, 100],
}

# %%
print("=" * 80)
print("VARIANT A: No Polynomial Features (Baseline)")
print("=" * 80)

grid_search_A = GridSearchCV(
    full_pipeline_A,
    param_grid_A,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2,
)
grid_search_A.fit(X_train, y_train_transformed)

print(f"\nBest MAE (CV): {-grid_search_A.best_score_:,.2f}")
print(f"Best params: {grid_search_A.best_params_}")

train_mae_A, val_mae_A = check_fit(grid_search_A, X_train, y_train, X_val, y_val, price_transformer)
print(f"\nTrain MAE: {train_mae_A:,.2f}")
print(f"Validation MAE: {val_mae_A:,.2f}")

# %% [markdown]
# ### Variant B: Polynomial Features Before Power Transform
#
# This variant tests whether feature interactions help:
# - Create polynomial features from RAW Mileage and Capacity first
# - Then apply power transform to Car_Age (and polynomial features pass through)
# - Everything else same as Variant A
#
# **Why this order?** Polynomials on raw features make sense (e.g., Mileage × Capacity).
# Polynomials on power-transformed features create numerical chaos (belive me i tried and i got an errro worse than the dummy, around 5mil)

# %%
full_pipeline_B = Pipeline(
    [
        ("preprocessing", preprocessing_pipeline_B),
        ("model", Ridge()),
    ]
)

param_grid_B = {
    "preprocessing__scaler": [StandardScaler(), RobustScaler(), None],
    "preprocessing__text_features__description_bow__max_features": [100, 1000, 5000],
    "preprocessing__text_features__edition_bow__max_features": [100, 1000, 5000],
    "preprocessing__poly_features__poly__degree": [1, 2],
    "preprocessing__poly_features__poly__interaction_only": [False, True],
    "preprocessing__feature_selection": [
        None,
        TruncatedSVD(n_components=100),
        TruncatedSVD(n_components=300),
    ],
    "model__alpha": [0.01, 0.1, 1, 10, 100],
}

# %%
print("\n" + "=" * 80)
print("VARIANT B: Polynomial Features Before Power Transform")
print("=" * 80)

grid_search_B = GridSearchCV(
    full_pipeline_B,
    param_grid_B,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2,
)
grid_search_B.fit(X_train, y_train_transformed)

print(f"\nBest MAE (CV): {-grid_search_B.best_score_:,.2f}")
print(f"Best params: {grid_search_B.best_params_}")

train_mae_B, val_mae_B = check_fit(grid_search_B, X_train, y_train, X_val, y_val, price_transformer)
print(f"\nTrain MAE: {train_mae_B:,.2f}")
print(f"Validation MAE: {val_mae_B:,.2f}")

# %% [markdown]
# ### Ridge Results Comparison
#
# Let's see which variant performs better. We'll use the winner's preprocessing pipeline for the other models.

# %%
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"\nVariant A (No Polynomials):")
print(f"  CV MAE: {-grid_search_A.best_score_:,.2f}")
print(f"  Train MAE: {train_mae_A:,.2f}")
print(f"  Val MAE: {val_mae_A:,.2f}")

print(f"\nVariant B (Poly Before Transform):")
print(f"  CV MAE: {-grid_search_B.best_score_:,.2f}")
print(f"  Train MAE: {train_mae_B:,.2f}")
print(f"  Val MAE: {val_mae_B:,.2f}")

if val_mae_A < val_mae_B:
    print(f"\n✓ Variant A is better by {val_mae_B - val_mae_A:,.2f}")
    best_ridge_model = grid_search_A
    best_preprocessing = preprocessing_pipeline_A
else:
    print(f"\n✓ Variant B is better by {val_mae_A - val_mae_B:,.2f}")
    best_ridge_model = grid_search_B
    best_preprocessing = preprocessing_pipeline_B

# %% [markdown]
# ## KNN Regressor
#
# K-Nearest Neighbors might capture non-linear price patterns that Ridge can't.
#
# We use RandomizedSearchCV with 50 iterations to keep it fast.

# %%
full_pipeline_knn = Pipeline(
    [
        ("preprocessing", best_preprocessing),
        ("model", KNeighborsRegressor()),
    ]
)

param_grid_knn = {
    "preprocessing__scaler": [StandardScaler(), RobustScaler()],
    "preprocessing__text_features__description_bow__max_features": [100, 500, 1000],
    "preprocessing__text_features__edition_bow__max_features": [100, 500, 1000],
    "preprocessing__feature_selection": [
        TruncatedSVD(n_components=50),
        TruncatedSVD(n_components=100),
        TruncatedSVD(n_components=200),
    ],
    "model__n_neighbors": [3, 5, 10, 20],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
}

# %%
print("\n" + "=" * 80)
print("KNN REGRESSOR")
print("=" * 80)

random_search_knn = RandomizedSearchCV(
    full_pipeline_knn,
    param_grid_knn,
    n_iter=50,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)
random_search_knn.fit(X_train, y_train_transformed)

print(f"\nBest MAE (CV): {-random_search_knn.best_score_:,.2f}")
print(f"Best params: {random_search_knn.best_params_}")

train_mae_knn, val_mae_knn = check_fit(random_search_knn, X_train, y_train, X_val, y_val, price_transformer)
print(f"\nTrain MAE: {train_mae_knn:,.2f}")
print(f"Validation MAE: {val_mae_knn:,.2f}")

# %% [markdown]
# ## Random Forest
#
# Random Forest is our ensemble approach.
# Random forests often work well "out of the box" but we'll still tune them.

# %%
full_pipeline_rf = Pipeline(
    [
        ("preprocessing", best_preprocessing),
        ("model", RandomForestRegressor(random_state=42)),
    ]
)

param_grid_rf = {
    "preprocessing__scaler": [None, StandardScaler()],
    "preprocessing__text_features__description_bow__max_features": [100, 500, 1000],
    "preprocessing__text_features__edition_bow__max_features": [100, 500, 1000],
    "preprocessing__feature_selection": [
        None,
        TruncatedSVD(n_components=100),
        TruncatedSVD(n_components=200),
    ],
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.5],
}

# %%
print("\n" + "=" * 80)
print("RANDOM FOREST")
print("=" * 80)

random_search_rf = RandomizedSearchCV(
    full_pipeline_rf,
    param_grid_rf,
    n_iter=50,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)
random_search_rf.fit(X_train, y_train_transformed)

print(f"\nBest MAE (CV): {-random_search_rf.best_score_:,.2f}")
print(f"Best params: {random_search_rf.best_params_}")

train_mae_rf, val_mae_rf = check_fit(random_search_rf, X_train, y_train, X_val, y_val, price_transformer)
print(f"\nTrain MAE: {train_mae_rf:,.2f}")
print(f"Validation MAE: {val_mae_rf:,.2f}")

# %% [markdown]
# ## Final Model Comparison
#
# Now we compare all models side-by-side to select the winner.

# %%
print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

results = {
    "Ridge (Variant A)": (grid_search_A, -grid_search_A.best_score_, train_mae_A, val_mae_A),
    "Ridge (Variant B)": (grid_search_B, -grid_search_B.best_score_, train_mae_B, val_mae_B),
    "KNN": (random_search_knn, -random_search_knn.best_score_, train_mae_knn, val_mae_knn),
    "Random Forest": (random_search_rf, -random_search_rf.best_score_, train_mae_rf, val_mae_rf),
}

for model_name, (model, cv_mae, train_mae, val_mae) in results.items():
    print(f"\n{model_name}:")
    print(f"  CV MAE: {cv_mae:,.2f}")
    print(f"  Train MAE: {train_mae:,.2f}")
    print(f"  Val MAE: {val_mae:,.2f}")

best_model_name = min(results.items(), key=lambda x: x[1][3])
print(f"Best model: {best_model_name[0]} with Val MAE: {best_model_name[1][3]:,.2f}")
best_model = best_model_name[1][0]

# %%
