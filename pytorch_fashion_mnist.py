import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from tqdm.auto import tqdm
from timeit import default_timer as timer

# This file will use PyTorch to train a model on the Fashion-MNIST data in order for us to compare our neural network we created from scratch to what PyTorch can do!

class PyTorchModel(nn.Module):
    '''
    Defines the model that we will use to predict the Fashion MNIST data
    '''
    def __init__(self,
                input_shape: int,
                hidden_units: int,
                output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn):
  '''
  Runs the train step for the train and testing loop
  '''
  train_loss, train_acc = 0, 0
  for batch, (X, y) in enumerate(data_loader):

    model.train()

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}%\n")

def test_step(model, data_loader, loss_fn, accuracy_fn):
  '''
  Runs the test step for the train and testing loop
  '''
  test_loss, test_acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for X_test, y_test in data_loader:

      test_pred = model(X_test)

      test_loss += loss_fn(test_pred, y_test)
      test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

  print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%\n")

def print_train_time(start: float, end: float):
    '''
    Calculates the train time when given start and end times.
    '''
    total_time = end - start
    print(f"Train time: {total_time:.3f} seconds")
    return total_time



# Load Fashion MNIST Data and put into train and test dataloaders

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=32,
                              shuffle=False)

class_names = train_data.classes


# Create PyTorch model

model = PyTorchModel(input_shape=784,
                        hidden_units=10,
                        output_shape=len(class_names))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)



# Train and test model

train_time_start = timer()

epochs = 5

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch+1}\n------")

  train_step(model=model,
             data_loader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn)
  test_step(model=model,
             data_loader=test_dataloader,
             loss_fn=loss_fn,
             accuracy_fn=accuracy_fn)

train_time_end = timer()

total_train_time_model_2 = print_train_time(start=train_time_start,
                                            end=train_time_end)