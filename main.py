from src.model import ClassificationModel
from src.data_handler import DataHandler
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchinfo import summary


data_handler = DataHandler(data_path='data/causal_dataset.csv')

data_handler.get_data_info()
print(data_handler.get_data_shape())
X_train, X_test, y_train, y_test = data_handler.get_data_tensors()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = ClassificationModel(input_size=X_train.shape[1], 
                            hidden_units=20, 
                            output_shape=1, 
                            dropout=True)

model_summary = summary(model, input_size=(X_train.shape[1], ))
print(model_summary)

EPOCHS = 100 
LEARNING_RATE = 0.001

# define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


st = timer()
# train model
for epoch in range(EPOCHS):
    model.train()

    # forward pass
    y_pred = model(X_train)
    
    # compute loss
    loss = criterion(y_pred, y_train.unsqueeze(dim=1))

    # backward pass
    loss.backward()

    # zero out gradients
    optimizer.zero_grad()

    # update weights
    optimizer.step()

    # print loss
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.3f}')

et = timer()

print(f'Time elapsed: {et-st:.2f} seconds')

# evaluate model
model.eval()
with torch.inference_mode():
    y_pred = model(X_test)
    y_pred = torch.sigmoid(y_pred)
    y_pred = torch.round(y_pred)
    acc = (y_pred == y_test.unsqueeze(dim=1)).sum() / y_test.shape[0]
    print(f'Accuracy: {acc.item():.3f}')
