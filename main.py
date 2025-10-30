# %% [markdown]
# # Load the dataset
# %%
import pandas as pd

# Load the dataset
df = pd.read_csv("dataset_vehicles.csv")

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
df.head()

# %%
df.info()

# %%
df[df.duplicated()]
# they posted their car twice. Nuke them :D

# %%
df.drop_duplicates().head()


# %% [markdown]
# hmm but this will drop  the duplicates that are duplicates for all columns. what if somebday marked his car with one column difference:?
#

# %%
df[df.duplicated(subset=["Description"], keep=False) & ~df.duplicated()].sort_values(
    by=["Description"]
)


# %%
df["Post_URL"]

# %% [markdown]
# ### Post url seems useless might drop it

# %%
df

# %%
df["Seller_name"].value_counts()

# %% [markdown]
# ### Seller name could be useful, maybe there is a correlation between names of big sellers and and car prices ?
# Would be worth it to Pick the biggest N sellers and make everybody else have an "Unknown Seller"  type ?... Hmmm

# %% [markdown]
# The lengths is 18938 and we cans ee that there are null fields in the fields: Edition (13908/18938 entries) and Body (17038/18938). Due to Edition having too many nulls we can't really drop them... So we might have to think of another way to handle this. Let's look at the value counts of Edition.

# %%
# how many sellers are seen more than 10 times
df[
    df["Seller_name"].isin(
        df["Seller_name"].value_counts()[df["Seller_name"].value_counts() > 3].index
    )
]


# %%
df["Edition"].value_counts()

# %%
df[df["Description"].str.contains("", na=True)]

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
df[
    df["Edition"].isin(
        df["Edition"].value_counts()[df["Edition"].value_counts() > 1].index
    )
]

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
df[df["Fuel"] == "Other fuel type"].head()

# %%
df["Capacity"].value_counts()

# %%
# after trying to clean this i got a problem where magically everything became Null :D
# so im now going back to see whetehr evertyhign follows teh format <NUMBER cc> :DDDDD

# %%
pattern = r"^\d+(\.\d+)?\s+cc$"
non_matching = df[~df["Capacity"].str.match(pattern, na=False)]
non_matching[["Capacity"]]


# %% [markdown]
# Okay yeah it's the commas that are the problem... I will just nuke them in the data cleaning layer

# %%
df.describe()

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

# %%
len(
    df[df.apply(lambda row: str(row["Price"]) in str(row["Description"]), axis=1)]
)  # check if the price is in a lot of descriptions

# %% [markdown]
# ## Data cleaning

# %% [markdown]
# Things to do:
#
# [ ] Convert Capacity to number by creating a transformer that removes " cc" at the end
#
# [ ] Nuke Seller_type because everyone is a premium member
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
                X[col] = (
                    X[col].astype(str).str.replace(pattern, "", regex=True).str.strip()
                )
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
                X[col] = X[col].astype(str).str.lower()
                X[col] = X[col].str.replace(url_pattern, "", regex=True).str.strip()
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
        existing_mappings = {
            old: new for old, new in self.rename_mapping.items() if old in X.columns
        }
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
        (
            "drop_columns",
            ColumnNukerTransformer(
                columns_to_drop=["Post_URL", "Title", "Sub_title", "Seller_type"]
            ),
        ),
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
        (
            "clean_text",
            TextCleanerTransformer(
                text_columns=["Edition", "Description", "Seller_name"]
            ),
        ),
        (
            "rename_columns",
            ColumnRenamerTransformer(
                rename_mapping={
                    "published_date": "Published_Date"
                }  # i just don't like the fact it is not following the pattern
            ),
        ),
        (
            "convert_datetime",
            DatetimeToUnixTransformer(datetime_columns=["Published_Date"]),
        ),
    ]
)

# %%
df_cleaned = cleaning_pipeline.fit_transform(df)

# %%
df_cleaned.head()

# %%
df_cleaned.info()

# %% [markdown]
# # Train-Test Split

# %%
from sklearn.model_selection import train_test_split

X = df_cleaned.drop("Price", axis=1)
y = df_cleaned["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

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
# 1. Relationship between Price and Brand
# 2. Relationship between Price and Model
# 3. Relationshhip between Price and Year
# 4. Relationship between Body and Price
# 5. Relationship between Seller_type and Price ( maybe premium sellers sell more expensive cars ? )
# 6. Relationship between Fuel type and Price (If cetain fuel types are more expensive on average then maybe a good feature would be capacity + fuel type?? )
# 7. Car age  to price
# 8. Mileage per year -# chat gpt suggested this one :D

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
df_train.describe()

# %%
from scipy.stats import skew

# https://www.geeksforgeeks.org/python/scipy-stats-skew-python/
# Highly skeweed values (over 1) need to be transformed.
skew(df_cleaned["Mileage"]), skew(df_cleaned["Price"])

# %%
plt.figure(figsize=(12, 6))
plt.scatter(df_train.index, df_train["Price"], alpha=0.6, s=20)
plt.xlabel("Index")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.show()

# %%
cols = ["Mileage", "Capacity", "Price"]

plt.figure(figsize=(10, 8))

for i, col in enumerate(cols, 1):
    plt.subplot(3, 1, i)
    plt.scatter(df_cleaned.index, df_cleaned[col], alpha=0.6)
    plt.title(f"{col} Outliers")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.grid(True)

plt.tight_layout()
plt.show()


# %% [markdown]
# There are a few outliers I need to clip. Seems like the point of clipping woiuld be around 1.0*1e8 or somewhere between 0.75 and 1 for price, 6000 for capacity and

# %%
top_brands = df_train["Brand"].value_counts().head(10).index
df_brand_price = df_train[df_train["Brand"].isin(top_brands)]

brand_data = [
    df_brand_price[df_brand_price["Brand"] == brand]["Price"] for brand in top_brands
]

fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(brand_data, tick_labels=top_brands, vert=False)
ax.set_xlabel("Price")
ax.set_title("Price Distribution by Top 10 Brands")
plt.tight_layout()
plt.show()

# %% [markdown]
# Seems like a few brands are pretty representative (given the brand you can get a  a good price) this would be suzuki, nissan and honda.
# Others like land rover don't really give as a lot of data on the price :D
#

# %%
top_models = df_train["Model"].value_counts().head(15).index
df_model_price = df_train[df_train["Model"].isin(top_models)]

model_data = [
    df_model_price[df_model_price["Model"] == model]["Price"] for model in top_models
]

fig, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(model_data, tick_labels=top_models, vert=False)
ax.set_xlabel("Price")
ax.set_title("Price Distribution by Top 15 Models")
plt.tight_layout()
plt.show()

# %% [markdown]
# Some do seem kinda representative ?

# %% [markdown]
# ## Car Age to Price

# %%
current_year = 2025
df_train["Car_Age"] = current_year - df_train["Year"]

plt.figure(figsize=(14, 6))
plt.scatter(df_train["Car_Age"], df_train["Price"], alpha=0.3)
plt.xlabel("Car Age (years)")
plt.ylabel("Price (Rs)")
plt.title("Price vs Car Age")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Seems like a logarithmic relationship. With a really heavy tail

# %%

from scipy.stats import yeojohnson

df_train["Car_Age_transformed"] = yeojohnson(df_train["Car_Age"])[0]

# %%

correlation = df_train["Car_Age_transformed"].corr(df_train["Price"])
plt.scatter(df_train["Car_Age_transformed"], df_train["Price"], alpha=0.3)
plt.show()
print(correlation)

# %% [markdown]
# Better. Not the best

# %%
# https://www.geeksforgeeks.org/machine-learning/powertransformer-in-scikit-learn/
df_train["Price_transformed"], lambda_price = yeojohnson(df_train["Price"])


# %%

correlation = df_train["Car_Age_transformed"].corr(df_train["Price_transformed"])
plt.scatter(df_train["Car_Age_transformed"], df_train["Price_transformed"], alpha=0.3)
plt.show()
print(correlation)

# %%

# %% [markdown]
# Hmm this seems pretty good.

# %%
correlation_features = [
    "Price",
    "Year",
    "Mileage",
    "Capacity",
]
corr = df_train[correlation_features].corr()
corr.style.background_gradient(cmap="coolwarm")

# %% [markdown]
# ## I will check the distribution of all the numeric columns to see which would really need transformation
#

# %%
correlation_features = ["Price", "Year", "Mileage", "Capacity"]

plt.figure(figsize=(10, 8))

for i, col in enumerate(correlation_features, 1):
    plt.subplot(2, 2, i)
    plt.hist(df_train[col], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)

plt.tight_layout()
plt.show()


# %% [markdown]
# Okay this is pretty skewed.... A power transformer for all fo them might fix the issue.

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## Rare Category Analysis

# %%
high_cardinality_cols = ["Brand", "Model", "Location", "Seller_name"]

for col in high_cardinality_cols:
    print(f"\n{col} - Total unique values: {df_train[col].nunique()}")
    value_counts = df_train[col].value_counts()
    rare_threshold = len(df_train) * 0.002
    rare_categories = value_counts[value_counts < rare_threshold]
    print(
        f"Rare categories (< 0.2% of data or < {rare_threshold:.0f} occurrences): {len(rare_categories)}"
    )
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

# %% [markdown]
# ## Preprocessing Pipeline

# %%
df_cleaned.info()

# %%
df_cleaned["Body"].value_counts()

# %% [raw]
# We need to
# [] Add the new transformed fields and drop the un transformed ones
# [] Scale the  data (try different scalers in the grid search)
# [] One hot encode the Fuel type
# [] Ordinal Encode Condition field, Used, Reconditioned, New
# [] One-hot encode Transmission field: Automatic, Manual, Tiptonic, Other transmission
# [] Impute "Body" field with constant "Unknown" or something similiar
# [] One-hot encode Body field: Hatchback,SUV / 4x4, Station wagon, MPV, CoupÃ©/Sports,Convertible
# [] Bag of words the Description: https://stackoverflow.com/questions/30653642/combining-bag-of-words-and-other-features-in-one-model-using-sklearn-and-pandas
# [] Bag of words the Edition: Edition (could try and use a small vocab for this one, i think it would be worth it) or shove it into grid search ?

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
preprocessing_pipeline = Pipeline(
    [
        ("add_car_age", CarAgeTransformer(current_year=2025)),
        (
            "target_encode",
            ColumnTransformer(
                [
                    (
                        "model_location_encoder",
                        TargetEncoder(
                            categories="auto", target_type="continuous", cv=5
                        ),
                        ["Model", "Location"],
                    )
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        ("frequency_encode", FrequencyEncoderTransformer(columns=["Seller_name"])),
        (
            "impute_body",
            ColumnTransformer(
                [
                    (
                        "body_imputer",
                        SimpleImputer(strategy="constant", fill_value="unknown"),
                        ["Body"],
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
                        OneHotEncoder(
                            drop="first", sparse_output=False, handle_unknown="ignore"
                        ),
                        ["Fuel"],
                    ),
                    (
                        "transmission_encoder",
                        OneHotEncoder(
                            drop="first", sparse_output=False, handle_unknown="ignore"
                        ),
                        ["Transmission"],
                    ),
                    (
                        "body_encoder",
                        OneHotEncoder(
                            drop="first", sparse_output=False, handle_unknown="ignore"
                        ),
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
                        CountVectorizer(
                            max_features=50, lowercase=True, stop_words="english"
                        ),
                        "Description",
                    ),
                    (
                        "edition_bow",
                        CountVectorizer(
                            max_features=30, lowercase=True, stop_words="english"
                        ),
                        "Edition",
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
        ),
        ("scaler", None),
    ]
)

# %%
# since we need to transform the price now
#  y_pred = price_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()
# is needed  after pred.

# %% [markdown]
# # Model Selection

# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
import warnings

warnings.filterwarnings("ignore")

# %%
full_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing_pipeline),
        ("model", LinearRegression()),
    ]
)

param_grid = {
    "preprocessing__scaler": [StandardScaler(), RobustScaler(), None],
    "preprocessing__text_features__description_bow__max_features": [100, 200],
    "preprocessing__text_features__edition_bow__max_features": [50, 60],
    "preprocessing__poly_features__poly__degree": [1, 2, 3],
    "preprocessing__poly_features__poly__interaction_only": [False, True],
    "model__alpha": [0.1, 0.5, 1],
}

# %%
price_transformer = PowerTransformer(method="yeo-johnson")
y_train_transformed = price_transformer.fit_transform(
    y_train.values.reshape(-1, 1)
).ravel()
y_val_transformed = price_transformer.transform(y_val.values.reshape(-1, 1)).ravel()

# %%
grid_search = RandomizedSearchCV(
    full_pipeline,
    param_grid,
    cv=2,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train_transformed)

# %%
print(f"Best MAE (CV): {-grid_search.best_score_:,.2f}")
print(f"Best params: {grid_search.best_params_}")
print(grid_search.best_estimator_.n_features_in_)


# %%
def check_fit(model, X_train, y_train, X_val, y_val, price_transformer):
    y_train_pred_transformed = model.predict(X_train)
    y_train_pred = price_transformer.inverse_transform(
        y_train_pred_transformed.reshape(-1, 1)
    ).ravel()

    y_val_pred_transformed = model.predict(X_val)
    y_val_pred = price_transformer.inverse_transform(
        y_val_pred_transformed.reshape(-1, 1)
    ).ravel()

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)

    return train_mae, val_mae


# %%
print(check_fit(grid_search, X_train, y_train, X_val, y_val, price_transformer))

# %% [markdown]
# # Debugging

# %%
y_train.describe(), y_val.describe()

# %%
y_val_pred_transformed = grid_search.predict(X_val)
y_val_pred = price_transformer.inverse_transform(
    y_val_pred_transformed.reshape(-1, 1)
).ravel()

pd.DataFrame(
    {
        "actual": y_val.values,
        "predicted": y_val_pred,
        "error": np.abs(y_val.values - y_val_pred),
    }
).describe()

# %%
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
