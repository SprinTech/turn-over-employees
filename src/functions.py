from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def encodage(df):
    numerical_cols=df.select_dtypes(exclude='object')
    numerical_transformer = SimpleImputer(strategy='median')

    categorical_cols=df.select_dtypes(include='object')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
        
    return df


def imputation(df):
    df = df.dropna(axis=0)
    return df