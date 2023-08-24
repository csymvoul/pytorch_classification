from src.model import ClassificationModel
from src.data_handler import DataHandler
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchinfo import summary

from torch.autograd import Variable


data_handler = DataHandler(data_path='data/causal_dataset.csv')

train_loader, test_loader = data_handler.get_dataloaders()

# data_handler.get_data_info()
# print(data_handler.get_data_shape())
# X_train, X_test, y_train, y_test = data_handler.get_data_tensors()
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = ClassificationModel(input_size=train_loader.dataset.tensors[0].shape[1], 
                            hidden_units=10, 
                            output_shape=1, 
                            dropout=True)

model_summary = summary(model, input_size=(train_loader.dataset.tensors[0].shape[1], ))
print(model_summary)

EPOCHS = 300
LEARNING_RATE = 0.001

# # define loss function and optimizer
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = { "loss": [], "accuracy": [], "loss_val": [], "accuracy_val": [] }

# train model
for epoch in range(EPOCHS):
    loss = 0
    for idx, (minibatch, target) in enumerate(train_loader):
        y_pred = model(Variable(minibatch))

        loss = criterion(y_pred, Variable(target.float()))

        # compute loss
        loss += criterion(y_pred, Variable(target.unsqueeze(dim=1)))
        prediction = [1 if x > 0.5 else 0 for x in y_pred.data.numpy()]
        correct = (prediction == target.numpy()).sum()

        # compute accuracy
        acc = (y_pred == target.unsqueeze(dim=1)).sum() / target.shape[0]

        # zero out gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    history["loss"].append(loss.data[0])
    history["accuracy"].append(100 * correct / len(prediction))
    
    # print loss
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {history["loss"][-1]:.3f}, Accuracy: {history["accuracy"][-1]:.3f}')


# st = timer()
# # train model
# for epoch in range(EPOCHS):
#     model.train()

#     # forward pass
#     y_pred = model(X_train)
    
#     # compute loss
#     loss = criterion(y_pred, y_train.unsqueeze(dim=1))

#     # backward pass
#     loss.backward()

#     # zero out gradients
#     optimizer.zero_grad()

#     # update weights
#     optimizer.step()

#     # print loss
#     if epoch % 10 == 0:
#         print(f'Epoch: {epoch+1}, Loss: {loss.item():.3f}')

# et = timer()

# print(f'Time elapsed: {et-st:.2f} seconds')

# # evaluate model
# model.eval()
# with torch.inference_mode():
#     y_pred = model(X_test)
#     y_pred = torch.sigmoid(y_pred)
#     y_pred = torch.round(y_pred)
#     acc = (y_pred == y_test.unsqueeze(dim=1)).sum() / y_test.shape[0]
#     print(f'Accuracy: {acc.item():.3f}')
