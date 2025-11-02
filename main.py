# %% [markdown]
# # Load the dataset
# %%
import pandas as pd

df = pd.read_csv("dataset_vehicles.csv")
rngs = 42

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
# Cross-validation will handle validation during hyperparameter tuning to prevent leakage.
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


class OutlierClipperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_config):
        self.clip_config = clip_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, bounds in self.clip_config.items():
            if col in X.columns:
                lower, upper = bounds
                if lower is not None and upper is not None:
                    X[col] = X[col].clip(lower=lower, upper=upper)
                elif upper is not None:
                    X[col] = X[col].clip(upper=upper)
                elif lower is not None:
                    X[col] = X[col].clip(lower=lower)
        return X


# After analysis, winsorization will actually remove valid candidates and should not be done.
class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, limits=(0.01, 0.01)):
        self.columns = columns
        self.limits = limits
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                lower_pct, upper_pct = self.limits
                self.lower_bounds_[col] = X[col].quantile(lower_pct)
                self.upper_bounds_[col] = X[col].quantile(1 - upper_pct)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns and col in self.lower_bounds_:
                X[col] = X[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
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
        #I later decided that winsorization will actually drop valid data...
        # ("winsorize", WinsorizeTransformer(columns=["Price", "Mileage", "Capacity"], limits=(0.01, 0.01))),
        (
            "clip_outliers",
            OutlierClipperTransformer(
                clip_config={
                    "Capacity": (None, 6200),
                    "Price": (None, 100000000),
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

# %% [markdown]
# Split training set into  training and test. We are ordering by Published_Date to prevent leakage.
# We don't want the model to "see the future".

# %%
df_cleaned = df_cleaned.sort_values("Published_Date").reset_index(drop=True)
X = df_cleaned.drop("Price", axis=1)
y = df_cleaned["Price"]
n = len(df_cleaned)
train_size = int(0.8 * n)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

df_train = df_cleaned.iloc[:train_size]
df_test = df_cleaned.iloc[train_size:]


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


def plot_distributions(df, columns, transformed=False, layout="auto"):
    n_cols = len(columns)

    if layout == "auto":
        if n_cols <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = n_cols, 1
    else:
        rows, cols = layout

    if rows == 2 and cols == 2:
        figsize = (10, 8)
    else:
        figsize = (8, 4 * rows)

    colors = ["steelblue", "seagreen", "indianred", "darkorange", "mediumpurple"] * (n_cols // 5 + 1)

    plt.figure(figsize=figsize)

    for i, (col, color) in enumerate(zip(columns, colors), 1):
        plt.subplot(rows, cols, i)
        plt.hist(df[col], bins=30, color=color, edgecolor="black", alpha=0.7)
        title = f"{col} Distribution" if not transformed else f"{col} Distribution"
        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel(col, fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_category_vs_target(
    df, category_col, target_col="Price", top_n=10, plot_type="boxplot", orientation="horizontal", figsize=(12, 6), color="steelblue", title=None
):
    top_categories = df[category_col].value_counts().head(top_n).index
    df_filtered = df[df[category_col].isin(top_categories)]

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "boxplot":
        category_data = [df_filtered[df_filtered[category_col] == cat][target_col] for cat in top_categories]

        if orientation == "horizontal":
            ax.boxplot(category_data, tick_labels=top_categories, vert=False)
            ax.set_xlabel(target_col)
            ax.set_ylabel(category_col)
        else:
            ax.boxplot(category_data, tick_labels=top_categories, vert=True)
            ax.set_xlabel(category_col)
            ax.set_ylabel(target_col)

    elif plot_type == "bar":
        means = df_filtered.groupby(category_col)[target_col].mean().sort_values(ascending=False)

        if orientation == "horizontal":
            ax.barh(means.index, means.values, color=color)
            ax.set_xlabel(f"Mean {target_col}")
            ax.set_ylabel(category_col)
        else:
            ax.bar(means.index, means.values, color=color)
            ax.set_xlabel(category_col)
            ax.set_ylabel(f"Mean {target_col}")

    if title is None:
        title = f"{target_col} Distribution by Top {top_n} {category_col}"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_scatter_analysis(df, x_cols, y_col=None, show_correlation=False, subplot_layout="vertical"):
    if isinstance(x_cols, str):
        x_cols = [x_cols]

    n_cols = len(x_cols)

    if subplot_layout == "vertical":
        rows, cols = n_cols, 1
        figsize = (10, 8) if n_cols <= 3 else (10, 4 * n_cols)
    else:
        rows = int(np.ceil(n_cols / 2))
        cols = 2 if n_cols > 1 else 1
        figsize = (10, 8)

    if n_cols == 1:
        figsize = (14, 6)

    plt.figure(figsize=figsize)

    for i, col in enumerate(x_cols, 1):
        if n_cols > 1:
            plt.subplot(rows, cols, i)

        if y_col is None:
            plt.scatter(df.index, df[col], alpha=0.6)
            plt.xlabel("Index")
            plt.ylabel(col)
            plt.title(col)
        else:
            plt.scatter(df[col], df[y_col], alpha=0.3)
            plt.xlabel(col)
            plt.ylabel(y_col)
            if n_cols == 1:
                plt.title(f"{y_col} vs {col}")
            else:
                plt.title(col)

            if show_correlation:
                correlation = df[col].corr(df[y_col])
                print(f"Correlation between {col} and {y_col}: {correlation:.3f}")

        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# %% [markdown]
# Using a copy to avoid polluting the training dataframe during EDA experiments

# %%
df_train_copy = df_train.copy()

# %%
df_train_copy.describe()

# %% [markdown]
# the std for price is huge! 7.935010e+06
# I need to look at whether this data is actually valid.
#

# %%
df_train_copy.sort_values("Price",ascending=False).head()


# %%
from scipy.stats import skew

# https://www.geeksforgeeks.org/python/scipy-stats-skew-python/
# Highly skeweed values (over 1) need to be transformed.
skew(df_train_copy["Mileage"]), skew(df_train_copy["Price"])

# %% [markdown]
# --------------------------------------
# Let's take a look at some outliers.
# Note: Although this analysis is done on df_train_copy in EDA. The actual clipping is applied earlier in the data cleaning pipeline.

# %%
cols = ["Mileage", "Capacity", "Price"]
plot_scatter_analysis(df_train_copy, cols)


# %%
df_train_copy.sort_values("Price", ascending=False).head()

# %%
df_train_copy.sort_values("Mileage", ascending=False).head()

# %%
df_train_copy.sort_values("Capacity", ascending=False).head()

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
plot_category_vs_target(df_train_copy, "Brand", "Price", top_n=10)

# %% [markdown]
# Seems like a few brands are pretty representative (given the brand you can get a  a good price) this would be suzuki, nissan and honda.
# Others like land rover don't really give as a lot of data on the price.
#

# %%
plot_category_vs_target(df_train_copy, "Model", "Price", top_n=15, figsize=(12, 8))

# %% [markdown]
# Some do seem kinda representative but nothing really strikes me here.

# %% [markdown]
# ## Car Age to Price

# %%
current_year = 2025
df_train_copy["Car_Age"] = current_year - df_train_copy["Year"]

plot_scatter_analysis(df_train_copy, "Car_Age", y_col="Price")

# %% [markdown]
# Seems like a logarithmic relationship. With a really heavy tail. Tranforming it with a power tranformation will help for the parametric models like linear regression and ridge.

# %%

from scipy.stats import yeojohnson

df_train_copy["Car_Age_transformed"] = yeojohnson(df_train_copy["Car_Age"])[0]

# %%

plot_scatter_analysis(df_train_copy, "Car_Age_transformed", y_col="Price", show_correlation=True)

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
lambda_price

# %% [markdown]
# lambda shows how close to log - sqrt the transformation is. 

# %%

plot_scatter_analysis(df_train_copy, "Car_Age_transformed", y_col="Price_transformed", show_correlation=True)

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
plot_distributions(df_train_copy, correlation_features)


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
correlation_features = [
    "Price_transformed",
    "Car_Age_transformed",
    "Mileage_transformed",
    "Capacity_transformed",
    "Year_transformed",
]
plot_distributions(df_train_copy, correlation_features, transformed=True)


# %% [markdown]
# Price is still skewed to the right but is def better

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
# ## Feature Engineering Exploration

# %%
df_train_copy["Mileage_per_Year"] = df_train_copy["Mileage"] / (df_train_copy["Car_Age"] + 1)
df_train_copy["Mileage_per_Year"].corr(df_train_copy["Price"])

# %% [markdown]
# Pretty bad correlation but it could be non linear. Let me plot it

# %%
plot_scatter_analysis(df_train_copy, "Mileage_per_Year", y_col="Price")

# %%
len(df_train_copy[df_train_copy.apply(lambda row: str(row["Price"]) in str(row["Description"]), axis=1)])  # check if the price is in a lot of descriptions

# %% [markdown]
# Interesting but not good enough for me...

# %% [markdown]
# Non linear relationship. Although linear regression will not be able to get it, random forest or knn might be able to pick it up.

# %%
from collections import Counter
import re

edition_text = " ".join(df_train_copy["Edition"].dropna().str.lower())
tokens = re.findall(r"\b[a-z0-9]+\b", edition_text)
common_words = Counter(tokens).most_common(50)
for word, count in common_words:
    print(f"{word}: {count}")


# %%


edition_lower = df_train_copy["Edition"].str.lower().fillna("")


edition_groups = {
    "sporty": ["sport", "gt", "gti", "rs", "amg", "turbo"],
    "luxury": ["premium", "superior", "limited", "option"],
    "base": ["grade", "s", "ex", "g"],
    "utility": ["wagon", "diesel", "auto"],
}


def classify_edition(text):
    for group, keywords in edition_groups.items():
        if any(k in text for k in keywords):
            return group
    return "other"


df_train_copy["Edition_Category"] = edition_lower.apply(classify_edition)


edition_price_means = df_train_copy.groupby("Edition_Category")["Price"].mean().sort_values(ascending=False)
print(edition_price_means)

plot_category_vs_target(
    df_train_copy,
    "Edition_Category",
    "Price",
    top_n=len(df_train_copy["Edition_Category"].unique()),
    plot_type="bar",
    orientation="vertical",
    figsize=(8, 4),
    color="teal",
)


# %% [markdown]
# This seems pretty good actually! But we do need to make it a bit more sophisticated as we don't want to just hardcode the words.
# What we can do is try and 'cluster' words from the edition column into groups. This way we should be able to do what we did above but in a more general way.
#
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html

# %%
cluster_sizes = df_train_copy["Edition_Category"].value_counts()
print(cluster_sizes)


# %% [markdown]
# I'm concerned that this is not good enough and there are
# 1. too many things falling into t he other category
# 2. too litle things falling into sporty luxurt and utility
#
# And the difference in the mean price is just because of the small amount of data for each of the categories

# %% [markdown]
# What if we train a model to do regression using the text

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
# We won't capture relationship between location closeness this way but it's simpler..
#
#
# ----------------------------------

# %% [markdown]
# ## Price Trend Analysis Over Time

# %%
df_train_copy["Published_Date_readable"] = pd.to_datetime(df_train_copy["Published_Date"], unit="s")
df_train_copy["Year_Month"] = df_train_copy["Published_Date_readable"].dt.to_period("M").astype(str)

plot_category_vs_target(df_train_copy, "Year_Month", "Price", top_n=len(df_train_copy["Year_Month"].unique()), plot_type="bar", figsize=(14, 6))

# %% [markdown]
# This is bad. There is no good correlation between the published_date and the price. I feel like our dataset is too small  for this.
# There is a lot of deviation month to month... And as you can see the months don't follow any particular order. 

# %%
price_corr_with_time = df_train_copy["Published_Date"].corr(df_train_copy["Price"])
print(f"Correlation between Published_Date and Price: {price_corr_with_time:.4f}")

plot_scatter_analysis(df_train_copy, "Published_Date", y_col="Price", show_correlation=True)

# %%
df_train["Body"].value_counts()

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
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        return np.append(input_features, "car_age")

    def transform(self, X):
        X = X.copy()
        X["Car_Age"] = self.current_year - X["Year"]

        # keeping year (not dropping it) even though they are very similar (I tested with and without year and with year leads to a 5% improvement)
        return X


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


# %% [markdown]
# Preprocessing Pipelines:
#
# Ridge Variant A: Simple power transforms (for comparison)
# Ridge Variant B: Polynomial + power transforms (best for Ridge based on experiments)
# KNN: Minimal transforms + aggressive dimensionality reduction (knn really doesn't seem to like a lot of features)
# Random Forest: Raw features (trees handle non-linearity naturally)

# %%
preprocessing_pipeline_ridge_A = Pipeline(
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
                    ("capacity_yj", PowerTransformer(method="yeo-johnson"), ["Capacity"]),
                    ("year_yj", PowerTransformer(method="yeo-johnson"), ["Year"]),
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
                        CountVectorizer(max_features=None, lowercase=True, stop_words="english"),
                        "Description",
                    ),
                    (
                        "edition_bow",
                        CountVectorizer(max_features=None, lowercase=True, stop_words="english"),
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
preprocessing_pipeline_ridge_B = Pipeline(
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
                [("poly", PolynomialFeatures(), ["Mileage", "Capacity"])],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        (
            "power_transform",
            ColumnTransformer(
                [
                    ("car_age_yj", PowerTransformer(method="yeo-johnson"), ["Car_Age"]),
                    ("year_yj", PowerTransformer(method="yeo-johnson"), ["Year"]),
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
                        CountVectorizer(max_features=None, lowercase=True, stop_words="english"),
                        "Description",
                    ),
                    (
                        "edition_bow",
                        CountVectorizer(max_features=None, lowercase=True, stop_words="english"),
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
preprocessing_pipeline_knn = Pipeline(
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
                        CountVectorizer(max_features=None, lowercase=True, stop_words="english"),
                        "Description",
                    ),
                    (
                        "edition_bow",
                        CountVectorizer(max_features=None, lowercase=True, stop_words="english"),
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

preprocessing_pipeline_rf = preprocessing_pipeline_knn

preprocessing_pipeline_gb = preprocessing_pipeline_ridge_A

# %% [markdown]
# # Model Selection
#
# After cleaning and preprocessing the data, we now need to find the best model for predicting car prices.
#
# All models will be tested with feature selection to find the optimal combination.
#
# We are using TimeSeriesSplit to prevent data leakage from the test set.
# (we shouldn't be able to see the future)

# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyRegressor



from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# %% [markdown]
# ### Let's Start with the DummyRegressor

# %%
dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train, y_train)

dummy_mae = mean_absolute_error(y_test, dummy_model.predict(X_test))
print(f"Dummy Model MAE: {dummy_mae:,.0f}")

# %% [markdown]
# We need to beat an error of `5,773,067`.

# %% [markdown]
# Warnings flood the console and it gets annoying. I know it's not a good practice to ignore all of them
# but f**k it.

# %%
import warnings

warnings.filterwarnings("ignore")


# %% [markdown]
# Still using TimeSeriesSplit to prevent leakadge (model shouldnt have training data with "future" car publishing dates)

# %%

# %%
def check_fit(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    return train_mae, test_mae


# %%
def create_pipeline(preprocessing, model, transform_target=True):
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", model)])
    if transform_target:
        return TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer(method="yeo-johnson"))
    return pipeline


def evaluate_model(model, X_train, y_train):
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    print(f"Best Params: {model.best_params_}")
    print(f"CV: {-model.best_score_:,.0f} | Train: {train_mae:,.0f}")
    return train_mae


# %% [markdown]
# ## Ridge Regression

# %% [markdown]
# Ridge A: Power transform features

# %%
full_pipeline_A = create_pipeline(preprocessing_pipeline_ridge_A, Ridge(), transform_target=True)

# param_grid_A = {
#     "regressor__preprocessing__scaler": [None],
#     "regressor__preprocessing__text_features__description_bow__max_features": [1, 500],
#     "regressor__preprocessing__text_features__edition_bow__max_features": [300],
#     "regressor__preprocessing__feature_selection": [PCA(n_components=100), PCA(n_components=150)],
#     "regressor__model__alpha": [5, 10, 15],
# }

param_grid_A = {
    "regressor__preprocessing__scaler": [None],
    "regressor__preprocessing__text_features__description_bow__max_features": [1],
    "regressor__preprocessing__text_features__edition_bow__max_features": [300],
    "regressor__preprocessing__feature_selection": [PCA(n_components=150)],
    "regressor__model__alpha": [5],
}

grid_search_A = GridSearchCV(full_pipeline_A, param_grid_A, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
grid_search_A.fit(X_train, y_train)

train_mae_A = evaluate_model(grid_search_A, X_train, y_train)

# %% [markdown]
# Ridge A results (from previous run):
# ```
# Best Params: {'regressor__model__alpha': 5, 'regressor__preprocessing__feature_selection': PCA(n_components=150), 'regressor__preprocessing__scaler': None, 'regressor__preprocessing__text_features__description_bow__max_features': 1, 'regressor__preprocessing__text_features__edition_bow__max_features': 300}
# CV: 1,260,946 | Train: 1,129,763
# ```

# %% [markdown]
# Ridge B: Polynomial + power transform

# %%
full_pipeline_B = create_pipeline(preprocessing_pipeline_ridge_B, Ridge(), transform_target=True)

# param_grid_B = {
#     "regressor__preprocessing__scaler": [None, StandardScaler()],
#     "regressor__preprocessing__text_features__description_bow__max_features": [1, 1000],
#     "regressor__preprocessing__text_features__edition_bow__max_features": [300, 500],
#     "regressor__preprocessing__poly_features__poly__degree": [2],
#     "regressor__preprocessing__poly_features__poly__interaction_only": [False],
#     "regressor__preprocessing__feature_selection": [PCA(n_components=100), None],
#     "regressor__model__alpha": [10, 15, 20],
# }

param_grid_B = {
    "regressor__preprocessing__scaler": [None],
    "regressor__preprocessing__text_features__description_bow__max_features": [1],
    "regressor__preprocessing__text_features__edition_bow__max_features": [500],
    "regressor__preprocessing__poly_features__poly__degree": [2],
    "regressor__preprocessing__poly_features__poly__interaction_only": [False],
    "regressor__preprocessing__feature_selection": [PCA(n_components=100)],
    "regressor__model__alpha": [10],
}

grid_search_B = GridSearchCV(full_pipeline_B, param_grid_B, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
grid_search_B.fit(X_train, y_train)



# %%
train_mae_B = evaluate_model(grid_search_B, X_train, y_train)

# %% [markdown]
# Ridge B results (from previous run):
# ```
# Best Params: {'regressor__model__alpha': 10, 'regressor__preprocessing__feature_selection': PCA(n_components=100), 'regressor__preprocessing__poly_features__poly__degree': 2, 'regressor__preprocessing__poly_features__poly__interaction_only': False, 'regressor__preprocessing__scaler': None, 'regressor__preprocessing__text_features__description_bow__max_features': 1, 'regressor__preprocessing__text_features__edition_bow__max_features': 500}
# CV: 1,258,197 | Train: 1,121,169
# ```

# %% [markdown]
# ## KNN

# %% [markdown]
# KNN: Basic preprocessing no transforms

# %%
full_pipeline_knn = create_pipeline(preprocessing_pipeline_knn, KNeighborsRegressor(), transform_target=True)

# param_grid_knn = {
#     "regressor__model__n_neighbors": [10, 15, 20],
#     "regressor__model__weights": ["uniform", "distance"],
#     "regressor__model__metric": ["euclidean"],
#     "regressor__preprocessing__scaler": [StandardScaler()],
#     "regressor__preprocessing__feature_selection": [PCA(n_components=50)],
#     "regressor__preprocessing__text_features__description_bow__max_features": [1, 300],
#     "regressor__preprocessing__text_features__edition_bow__max_features": [400],
# }

param_grid_knn = {
    "regressor__model__n_neighbors": [10],
    "regressor__model__weights": ["distance"],
    "regressor__model__metric": ["euclidean"],
    "regressor__preprocessing__scaler": [StandardScaler()],
    "regressor__preprocessing__feature_selection": [PCA(n_components=50)],
    "regressor__preprocessing__text_features__description_bow__max_features": [1],
    "regressor__preprocessing__text_features__edition_bow__max_features": [400],
}

grid_search_knn = GridSearchCV(full_pipeline_knn, param_grid_knn, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
grid_search_knn.fit(X_train, y_train)

train_mae_knn = evaluate_model(grid_search_knn, X_train, y_train)

# %% [markdown]
# This is defenetly overfitting but I do feel like there are better models for this so I will leave it as is and
# not tune it too much.

# %% [markdown]
# ## Random Forest

# %% [markdown]
# Random Forest: No transforms trees handle non-linearity

# %%
full_pipeline_rf = create_pipeline(preprocessing_pipeline_rf, RandomForestRegressor(random_state=rngs), transform_target=True)

# param_grid_rf = {
#     "regressor__preprocessing__text_features__description_bow__max_features": [1, 500],
#     "regressor__preprocessing__text_features__edition_bow__max_features": [1],
#     "regressor__preprocessing__scaler": [None],
#     "regressor__preprocessing__feature_selection": [None],
#     "regressor__model__n_estimators": [100, 150],
#     "regressor__model__max_depth": [12, 15],
#     "regressor__model__min_samples_split": [10, 20],
#     "regressor__model__min_samples_leaf": [5, 10],
#     "regressor__model__max_features": ["sqrt"],
#     "regressor__model__bootstrap": [True],
# }

param_grid_rf = {
    "regressor__preprocessing__text_features__description_bow__max_features": [1],
    "regressor__preprocessing__text_features__edition_bow__max_features": [1],
    "regressor__preprocessing__scaler": [None],
    "regressor__preprocessing__feature_selection": [None],
    "regressor__model__n_estimators": [100],
    "regressor__model__max_depth": [12],
    "regressor__model__min_samples_split": [20],
    "regressor__model__min_samples_leaf": [10],
    "regressor__model__max_features": ["sqrt"],
    "regressor__model__bootstrap": [True],
}

grid_search_rf = GridSearchCV(full_pipeline_rf, param_grid_rf, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

train_mae_rf = evaluate_model(grid_search_rf, X_train, y_train)

# %% [markdown]
# Random Forest results (from previous run):
# ```
# CV: 1,372,620 | Train: 1,008,912
# ```

# %% [markdown]
# ## Gradient Boosting

# %% [markdown]
# GB: With power-transformed target

# %%
full_pipeline_gb = Pipeline(
    [
        ("drop_published_date", ColumnNukerTransformer(columns_to_drop=[])),
        ("preprocessing", preprocessing_pipeline_gb),
        ("model", GradientBoostingRegressor(random_state=rngs)),
    ]
)
full_pipeline_gb = TransformedTargetRegressor(regressor=full_pipeline_gb, transformer=PowerTransformer(method="yeo-johnson"))

# param_grid_gb = {
#     "regressor__drop_published_date__columns_to_drop": [None, ["Published_Date"]],
#     "regressor__preprocessing__text_features__description_bow__max_features": [1, 1000],
#     "regressor__preprocessing__text_features__edition_bow__max_features": [500],
#     "regressor__preprocessing__scaler": [None],
#     "regressor__preprocessing__feature_selection": [
#         None,
#     ],
#     "regressor__model__n_estimators": [200, 250],
#     "regressor__model__max_depth": [6, 7, 8],
#     "regressor__model__learning_rate": [0.05, 0.07],
#     "regressor__model__subsample": [0.8],
#     "regressor__model__min_samples_split": [5],
#     "regressor__model__min_samples_leaf": [3, 5],
#     "regressor__model__max_features": [None, "sqrt"],
# }

param_grid_gb = {
    "regressor__drop_published_date__columns_to_drop": [["Published_Date"]],
    "regressor__preprocessing__text_features__description_bow__max_features": [1],
    "regressor__preprocessing__text_features__edition_bow__max_features": [500],
    "regressor__preprocessing__scaler": [None],
    "regressor__preprocessing__feature_selection": [None],
    "regressor__model__n_estimators": [250],
    "regressor__model__max_depth": [7],
    "regressor__model__learning_rate": [0.07],
    "regressor__model__subsample": [0.8],
    "regressor__model__min_samples_split": [5],
    "regressor__model__min_samples_leaf": [3],
    "regressor__model__max_features": [None],
}

grid_search_gb = GridSearchCV(full_pipeline_gb, param_grid_gb, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, y_train)

train_mae_gb = evaluate_model(grid_search_gb, X_train, y_train)

# %% [markdown]
# Gradient Boosting results (from previous run):
# ```
# Best Params: {'regressor__drop_published_date__columns_to_drop': ['Published_Date'], 'regressor__model__learning_rate': 0.07, 'regressor__model__max_depth': 7, 'regressor__model__max_features': None, 'regressor__model__min_samples_leaf': 3, 'regressor__model__min_samples_split': 5, 'regressor__model__n_estimators': 250, 'regressor__model__subsample': 0.8, 'regressor__preprocessing__feature_selection': None, 'regressor__preprocessing__scaler': None, 'regressor__preprocessing__text_features__description_bow__max_features': 1, 'regressor__preprocessing__text_features__edition_bow__max_features': 500}
# CV: 848,807 | Train: 561,231
# ```

# %%

print("Train cars:")
print(f"  Mean Year: {df_train['Year'].mean()}")
print(f"  Mean Mileage: {df_train['Mileage'].mean()}")
print(f"  Mean Price: {df_train['Price'].mean()}")


print("\nTest cars:")
print(f"  Mean Year: {df_test['Year'].mean()}")
print(f"  Mean Price: {df_test['Price'].mean()}")


print(f"\nMean Published_Date (Train): {pd.Timestamp(df_train['Published_Date'].mean(), unit='s')}")
print(f"Mean Published_Date (Test): {pd.Timestamp(df_test['Published_Date'].mean(), unit='s')}")

print(f"\nPrice difference: {df_test['Price'].mean() / df_train['Price'].mean()}")

# %% [markdown]
# ## Final Model Comparison

# %%
print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

results = {
    "Ridge (Variant A)": (grid_search_A, -grid_search_A.best_score_, train_mae_A),
    "Ridge (Variant B)": (grid_search_B, -grid_search_B.best_score_, train_mae_B),
    "KNN": (grid_search_knn, -grid_search_knn.best_score_, train_mae_knn),
    "Random Forest": (grid_search_rf, -grid_search_rf.best_score_, train_mae_rf),
    "Gradient Boosting": (grid_search_gb, -grid_search_gb.best_score_, train_mae_gb),
}

for model_name, (model, cv_mae, train_mae) in results.items():
    print(f" {model_name}:")
    print(f"  CV MAE: {cv_mae:,.2f}")
    print(f"  Train MAE: {train_mae:,.2f}")

best_model_name = min(results.items(), key=lambda x: x[1][1])
print(f"\nBest model (by CV): {best_model_name[0]} with CV MAE: {best_model_name[1][1]:,.2f}")
best_model = best_model_name[1][0]

# %% [markdown]
# # Evaluate the best model on the test set

# %%
print(f"\nEvaluating best model on test set...")
test_mae = mean_absolute_error(y_test, best_model.predict(X_test))
train_mae = mean_absolute_error(y_train, best_model.predict(X_train))
print(f"\nFinal Results for {best_model_name[0]}:")
print(f"  Train MAE: {train_mae:,.2f}")
print(f"  Test MAE: {test_mae:,.2f}")


# %% [markdown]
# ## What is happening why are our errors so big ? (and why is there a such a big difference between train and test ?)
#
# I do feel like the main culprit is the difference between how we have created the Train set and test set.
#
# using a timeseries split resulted in training and test having vastly different means. as we can see below.
# There is a difference in the mean price.
#
# I do believe this is  the cause of the big error we are finding. Our models are just not properly picking up the price trends.

# %%
print(f"Train: {df_train['Published_Date'].min()} to {df_train['Published_Date'].max()}")
print(f"Test: {df_test['Published_Date'].min()} to {df_test['Published_Date'].max()}")

print(f"Mean prices:")
print(f"Train: {df_train['Price'].mean():,.0f}")
print(f"Test: {df_test['Price'].mean():,.0f}")

# %% [markdown]
# Seen that we have such a big differene in the mean between the train and test it explains the inaccuracy

# %% [markdown]
# # What can we do?

# %% [markdown]
# We could:
# 1. Use shuffle split instead of TimeSeries. This will give us a better distributaion of the price but the model would be useless.
#    What do we care about a model  that can only predict past price ?
# 2. We could do some pretty extreme clipping in the hopes that this will trim the test set (again we are leaking by having this knowedge) and we are also clipping too much. I can argue that all those really high prices (mainly of Range Rovers) are competely valid. RangeRovers are new and an average range rover of that model does go for that price...
#
#
# So both solutions are not suitable.
#

# %% [markdown]
# # Conclusion

# %% [markdown]
# ------------------
# The error although seemingly big is actually not that bad when you compare it with the dummy

# %%
dummy_mae/test_mae

# %% [markdown]
# We have an over 3.5X improvement over the dummy.
