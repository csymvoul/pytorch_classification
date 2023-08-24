import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch 
from torch.utils.data import Dataset, DataLoader

class DataHandler():
    def __init__(self, 
                 data_path: str, 
                 BATCH_SIZE : int = 32) -> None:
        self.data_path = data_path
        self.data = self._load_data()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.BATCH_SIZE = BATCH_SIZE

    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)
    
    def _split_data(self) -> None: 
        x = self.data.drop('class', axis=1)
        y = self.data['class']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    def _scale_data(self) -> None:
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
    
    def _get_data_tensors(self): 
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def get_dataloaders(self): 
        self._split_data()
        self._scale_data()
        self._get_data_tensors()
        train_dataset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        test_dataset = torch.utils.data.TensorDataset(self.x_test, self.y_test)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        return train_dataloader, test_dataloader

    # def _preprocess_data(self) -> pd.DataFrame:
    #     self.data = self.data.dropna()
    #     self.data = self.data.drop_duplicates()
    #     self.data = self.data.reset_index(drop=True)
    #     return self.data

    # def get_data_shape(self) -> Tuple[int, int]:
    #     return self.data.shape
    
    # def get_data_columns(self) -> list:
    #     return self.data.columns.tolist()
    
    # def get_data_info(self) -> None:
    #     print(f'Data shape: {self.get_data_shape()}')
    #     print(f'Data columns: {self.get_data_columns()}')
    
    # def _split_train_test(self)  -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     self.data = self._preprocess_data()
    #     X = self.data.drop('class', axis=1)
    #     y = self.data['class']
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
    #                                                                             y, 
    #                                                                             test_size=0.2, 
    #                                                                             random_state=42, 
    #                                                                             shuffle=True)
    #     self._regularize_tensors()
    #     return self.X_train, self.X_test, self.y_train, self.y_test

    # def get_data_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     self._split_train_test()
    #     self.X_train = torch.tensor(self.X_train.values, dtype=torch.float32)
    #     self.X_test = torch.tensor(self.X_test.values, dtype=torch.float32)
    #     self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
    #     self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)
    #     return self.X_train, self.X_test, self.y_train, self.y_test

    # def _regularize_tensors(self) -> None: 
    #     self.X_train = (self.X_train - self.X_train.mean()) / self.X_train.std()
    #     self.X_test = (self.X_test - self.X_test.mean()) / self.X_test.std()
    #     scaler = StandardScaler()
    #     self.X_train = scaler.fit_transform(self.X_train)
    #     self.X_test = scaler.transform(self.X_test)