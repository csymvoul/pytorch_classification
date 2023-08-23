from src.model import ClassificationModel
from src.data_handler import DataHandler
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as timer


data_handler = DataHandler(data_path='data/dataset.csv')
train, val, test = data_handler.get_data(train_size=0.7, val_size=0.15)

data_handler.get_data_info()
print(data_handler.get_data_shape())
train_X, train_y, test_X, test_y = data_handler.split_train_test()
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = ClassificationModel(input_size=10, hidden_units=20, output_shape=2, dropout=True)

EPOCHS = 10 
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# create data loader


# train model
for epoch in tqdm(range(EPOCHS)):
    model.train()
    for batch, (train_X, train_y) in enumerate(train_loader):
    # forward pass
    y_pred = model(train_X)
    loss = criterion(y_pred, train_y)
