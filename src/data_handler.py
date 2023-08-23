import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

class DataHandler():
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)

    def _preprocess_data(self) -> pd.DataFrame:
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()
        self.data = self.data.reset_index(drop=True)
        return self.data

    def _split_data(self, train_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data = self._preprocess_data()
        # split into train and test
        train, test = train_test_split(self.data, test_size=1-train_size, random_state=42, shuffle=True)
    
    def get_data(self, train_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._split_data(train_size, val_size)

    def get_data_shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    def get_data_columns(self) -> list:
        return self.data.columns.tolist()
    
    def get_data_info(self) -> None:
        print(f'Data shape: {self.get_data_shape()}')
        print(f'Data columns: {self.get_data_columns()}')
    
    def split_train_test(self)  -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data = self._preprocess_data()
        X = self.data.drop('class', axis=1)
        y = self.data['class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
                                                                                y, 
                                                                                test_size=0.2, 
                                                                                random_state=42, 
                                                                                shuffle=True)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
