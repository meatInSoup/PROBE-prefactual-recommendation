import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils import helpers


class DummyScaler(StandardScaler):
    def transform(self, X):
        return X.to_numpy()

    def inverse_transform(self, X):
        return X


# class Transformer:

#     def __init__(self, data, numerical, scale=True):
#         self.data = data
#         self.empty_df = pd.DataFrame(columns=data.columns)

#         self.num_name = numerical
#         self.cat_name = list(data.columns.difference(numerical))

#         numeric_transformer = StandardScaler(with_std=scale, with_mean=scale)

#         categorical_transformer = OneHotEncoder(handle_unknown='ignore')

#         self.encs = []
#         if len(self.num_name) > 0:
#             self.encs.append(('num', numeric_transformer, self.num_name))
#         if len(self.cat_name) > 0:
#             self.encs.append(('cat', categorical_transformer, self.cat_name))

#         self.transformer = ColumnTransformer(transformers=self.encs)

#         self.transformer.fit(data)

#         self.cat_indices = []
#         self.n_num_features_out = len(self.num_name)

#         if len(self.num_name) > 0:
#             self.enc_num = self.transformer.named_transformers_['num']
#         if len(self.cat_name) > 0:
#             self.enc_cat = self.transformer.named_transformers_['cat']

#             self.n_cat_features_out = len(self.enc_cat.get_feature_names())
#             self.cat_indices = list(range(self.n_num_features_out,
#                                           self.n_num_features_out + self.n_cat_features_out))
#         else:
#             self.n_cat_features_out = 0

#     def transform(self, X):
#         return self.transformer.transform(X)

#     def inverse_transform(self, X):
#         df = self.empty_df.copy()
#         n_num = len(self.num_name)

#         if n_num > 0:
#             X_num = X[..., :n_num]
#             inv_num = self.enc_num.inverse_transform(X_num)
#             df[self.num_name] = inv_num

#         if len(self.cat_name) > 0:
#             X_cat = X[..., n_num:]
#             inv_cat = self.enc_cat.inverse_transform(X_cat)
#             df[self.cat_name] = inv_cat

#         return df


class Transformer:
    def __init__(self, data, numerical, scale=True):
        self.data = data
        self.empty_df = pd.DataFrame(columns=data.columns)

        self.num_name = numerical
        self.cat_name = list(data.columns.difference(numerical))

        # Standard Scaler for Numerical Features
        numeric_transformer = StandardScaler(with_std=True, with_mean=False)
        # Label Encoder for Categorical Features
        categorical_transformer = OrdinalEncoder()

        # Label Encoder for Categorical Features
        # self.label_encoders = {}
        # data_encoded = self.data.copy()
        # for col in self.cat_name:
        #     le = LabelEncoder()
        #     data_encoded[col] = le.fit_transform(data[col])  # Encode categorical values in-place
        #     print(data_encoded.head(1))
        #     self.label_encoders[col] = le  # Store the encoder

        # self.encs = []
        # if len(self.num_name) > 0:
        #     self.encs.append(('num', numeric_transformer, self.num_name))
        
        self.encs = []
        if len(self.num_name) > 0:
            self.encs.append(('num', numeric_transformer, self.num_name))
        if len(self.cat_name) > 0:
            self.encs.append(('cat', categorical_transformer, self.cat_name))

        self.transformer = ColumnTransformer(transformers=self.encs)
        self.transformer.fit(data)
        
        if len(self.num_name) > 0:
            self.enc_num = self.transformer.named_transformers_['num']
        if len(self.cat_name) > 0:
            self.enc_cat = self.transformer.named_transformers_['cat']

    def transform(self, X):
        # transformed_X = self.transformer.transform(X)
        # df = pd.DataFrame(transformed_X, columns=self.num_name)  # Convert numerical back to DataFrame
        # for col in self.cat_name:
        #     df[col] = X[col].values  # Keep categorical values as-is
        # return df
        return self.transformer.transform(X)

    def inverse_transform(self, X):
        df = self.empty_df.copy()
        df[self.num_name] = self.transformer.named_transformers_['num'].inverse_transform(X[self.num_name])

        # Convert numeric categories back to original labels
        for col in self.cat_name:
            df[col] = self.label_encoders[col].inverse_transform(X[col].astype(int))

        return df



def get_transformer(dataset_name):
    # dataset_name = dataset_name.replace('shift', '')
    dataset, numerical = helpers.get_full_dataset(dataset_name)
    dataset = dataset.drop('label', axis=1)

    # transformer = Transformer(dataset, numerical, ('synthesis' not in dataset_name))
    transformer = Transformer(dataset, numerical)
    return transformer
