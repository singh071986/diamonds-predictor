import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Analyzer:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def _drop_unnamed_columns(self, df=None):
        """Drop auto-generated index columns like 'Unnamed: 0'.

        If df is provided, return a cleaned copy.
        Otherwise, clean self.data in place and return it.
        """
        if df is None:
            target = self.data
            in_place = True
        else:
            target = df.copy()
            in_place = False

        unnamed_cols = [c for c in target.columns if str(c).lower().startswith('unnamed')]
        if unnamed_cols:
            target.drop(columns=unnamed_cols, inplace=True, errors='ignore')

        if in_place:
            self.data = target
        return target

    def read_dataset(self, file_path):
        """Read/replace dataset from a CSV path."""
        self.data = pd.read_csv(file_path)
        return self.data

    def describe(self):
        """Return dataframe statistical description."""
        return self.data.describe(include='all')

    def show_info(self):
        print(f"show_info::{self.data.info()}")

    def show_head(self):
        print(self.data.head())

    def drop_columns(self, columns):
        """Drop columns if they exist in the dataframe."""
        if not columns:
            return
        existing = [c for c in columns if c in self.data.columns]
        if existing:
            self.data.drop(columns=existing, inplace=True)

    def shuffle(self, random_state=42):
        """Shuffle the dataset rows deterministically by default."""
        self.data = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return self.data

    def Shuffle(self, random_state=42):
        """Backward-compatible alias matching project spec naming."""
        return self.shuffle(random_state=random_state)

    def drop_missing_data(self):
        """Remove rows with missing values."""
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        return self.data



    def preprocess_data(self):
        """
        Preprocesses the dataset by dropping missing data, encoding categorical features, and scaling numerical features.
        """
        self.drop_missing_data()
        self._drop_unnamed_columns()

        # Encode categorical features
        le = LabelEncoder()
        for column in self.data.select_dtypes(include=['object', 'string']).columns:
            self.data[column] = le.fit_transform(self.data[column])

        # Scale numerical features
        scaler = StandardScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        self.shuffle()
        print("Data preprocessing completed.")


    def save_cleaned_data(self, output_path="cleaned_diamonds.csv"):
        # Ensure saved cleaned dataset does not contain auto-generated index columns.
        self._drop_unnamed_columns()
        self.data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}.")




    def encode_features(self, columns):
        """
        Encodes categorical features into numerical values.
        :param columns: List of column names to encode.
        """
        le = LabelEncoder()
        for column in columns:
            self.data[column] = le.fit_transform(self.data[column])
        print(f"Columns {columns} encoded.")

    def encode_label(self, column):
        """
        Encodes the target label into numerical values.
        :param column: Name of the target column to encode.
        """
        le = LabelEncoder()
        self.data[column] = le.fit_transform(self.data[column])
        print(f"Label column '{column}' encoded.")


    def sample(self, reduction_factor):
        """
        Return a randomly sampled dataset using a fraction in (0, 1].
        :param reduction_factor: Fraction of data to keep.
        """
        if reduction_factor <= 0.0 or reduction_factor > 1.0:
            raise ValueError("reduction_factor must be in the range (0, 1].")
        return self.data.sample(frac=reduction_factor, random_state=42).reset_index(drop=True)

    def plot_correlation_matrix(self, save_path=None, show=True):
        """Plot the correlation matrix of the dataset."""
        # Encode non-numeric columns to numeric values
        encoded_data = self._drop_unnamed_columns(self.data)
        for column in encoded_data.select_dtypes(include=['object', 'string']).columns:
            encoded_data[column] = encoded_data[column].astype('category').cat.codes

        # Calculate the correlation matrix
        corr = encoded_data.corr()

        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_correlationMatrix(self, save_path=None, show=True):
        """Alias matching project requirement naming."""
        return self.plot_correlation_matrix(save_path=save_path, show=show)

    def plot_pairPlot(self, columns=None, save_path=None, show=True):
        """
        Plots pairwise relationships in the dataset.
        """
        base_data = self._drop_unnamed_columns(self.data)
        pair_data = base_data if columns is None else base_data[columns]
        pair_grid = sns.pairplot(pair_data)
        pair_grid.fig.suptitle("Pair Plot of the Dataset", y=1.02)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            pair_grid.fig.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close(pair_grid.fig)

    def plot_histograms_numerical(self, columns=None, save_path=None, show=True):
        """
        Plots histograms for all numerical columns in the dataset.
        """
        base_data = self._drop_unnamed_columns(self.data)
        numerical_data = base_data.select_dtypes(include=['number']) if columns is None else base_data[columns]
        numerical_data.hist(bins=20, figsize=(15, 10))
        plt.suptitle("Histograms of Numerical Columns")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def Plot_histograms_numerical(self, columns=None, save_path=None, show=True):
        """Alias matching project requirement naming."""
        return self.plot_histograms_numerical(columns=columns, save_path=save_path, show=show)

    def plot_histograms_categorical(self, columns=None, save_dir=None, show=True):
        """
        Plots bar charts for all categorical columns in the dataset.
        """
        base_data = self._drop_unnamed_columns(self.data)
        categorical_columns = (
            #base_data.select_dtypes(include=['object', 'string', 'category']).columns
            base_data.select_dtypes(exclude=['number']).columns if columns is None else columns
            if columns is None
            else columns
        )
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        for column in categorical_columns:
            base_data[column].value_counts().plot(kind='bar')
            plt.title(f"Bar Chart for {column}")
            if save_dir:
                plt.tight_layout()
                plt.savefig(f"{save_dir}/{column}_hist.png")
            if show:
                plt.show()
            else:
                plt.close()

    def Plot_histograms_categorical(self, columns=None, save_dir=None, show=True):
        """Alias matching project requirement naming."""
        return self.plot_histograms_categorical(columns=columns, save_dir=save_dir, show=show)

    def plot_boxPlot(self, column, save_path=None, show=True):
        base_data = self._drop_unnamed_columns(self.data)
        sns.boxplot(data=base_data, y=column)
        plt.title(f"Box Plot for {column}")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            plt.tight_layout()
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def Plot_boxPlot(self, column, save_path=None, show=True):
        """Alias matching project requirement naming."""
        return self.plot_boxPlot(column=column, save_path=save_path, show=show)

