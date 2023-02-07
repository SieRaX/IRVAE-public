import torch
import torchvision
import time
from models.modules import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = torchvision.datasets.MNIST('./', train=True, download=True)
test_dataset = torchvision.datasets.MNIST('./', train=False)

x_train_orig = train_dataset.train_data
y_train_orig = train_dataset.train_labels
x_test_orig = test_dataset.test_data
y_test_orig = test_dataset.test_labels
print('x_train shape : ', x_train_orig.shape)
print('y_train shape : ', y_train_orig.shape)
print('x_test  shape : ', x_test_orig.shape)
print('y_test  shape : ', y_test_orig.shape)

# normalization
x_train = (x_train_orig[10000:] / 255.).to(device=device)
y_train = y_train_orig[10000:].to(device=device)
x_val = (x_train_orig[:10000] / 255.).to(device=device)
y_val = y_train_orig[:10000].to(device=device)
x_test = x_test_orig / 255.
y_test = y_test_orig.to(device=device)
 
print('x_train shape : ', x_train.shape)
print('y_train shape : ', y_train.shape)
print('x_train shape : ', x_val.shape)
print('y_train shape : ', y_val.shape)
print('x_test  shape : ', x_test.shape)
print('y_test  shape : ', y_test.shape)

trainingset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(trainingset, batch_size=32, shuffle=True)

m1 = Net()

m1.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m1.parameters(), lr=0.001)

for epoch in range(10):
    start = time.time()
    total_loss = 0
 
    for xb, yb in train_loader:
        #xb, yb = torch.autograd.Variable(xb), torch.autograd.Variable(yb)
 
        pred = m1(xb)
        loss = criterion(pred, yb)
 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        total_loss += loss.item()
    
    with torch.no_grad():
        pred = m1(x_train)
        acc = pred.data.max(1)[1].eq(y_train.data).sum()/len(x_train) * 100
        loss = criterion(pred, y_train)
    print(f"{time.time() - start} sec - loss : {loss} / acc : {acc}")

pred = m1(x_test.to(device=device))
acc = pred.data.max(1)[1].eq(y_test.to(device=device).data).sum()/len(x_test) * 100
loss = criterion(pred, y_test.to(device=device))
print(f"Test loss : {loss} / acc : {acc}")

# Save model
PATH = "models/saved_model/simple_linear.pt"
torch.save(m1.state_dict(), PATH)