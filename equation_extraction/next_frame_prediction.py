import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 1000
batch_size = 50
learning_rate = 0.001

inputs = torch.load('inputs.pt')
outputs = torch.load('outputs.pt') * 1000

training_set_percentage = 0.5
testing_set_percentage = 1 - training_set_percentage

training_indices = np.random.choice(np.arange(len(inputs)), int(len(inputs) * training_set_percentage), replace = False)
testing_indices = np.array(list(set(range(len(inputs))) - set(training_indices)))

training_set_inputs = inputs[training_indices,:2,:,:]
training_set_outputs = outputs[training_indices,:]
testing_set_inputs = inputs[testing_indices,:2,:,:]
testing_set_outputs = outputs[testing_indices,:]

batches = []
total_training_indices = np.arange(len(training_set_inputs))
for batch in range((len(training_set_inputs) / batch_size) + 1):
    if len(total_training_indices) >= batch_size:
        batch_indices = np.random.choice(total_training_indices, batch_size, replace = False)
    else:
        batch_indices = total_training_indices
    total_training_indices = np.array(list(set(total_training_indices) - set(batch_indices)))
    batches.append((training_set_inputs[batch_indices, :, :, :], training_set_outputs[batch_indices, :]))

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(9600, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)        
        
    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    i = 0
    for (images, positions) in batches:
        i += 1
        x = images.to(device)
        y = positions.to(device)
        
        pred_y = model(images)
        loss = torch.sqrt(criterion(pred_y, y))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i, len(batches), loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    average_testing_loss = 0
    x = testing_set_inputs.to(device)
    y = testing_set_outputs.to(device)
    pred_y = model(x)
    loss = torch.sqrt(criterion(pred_y, y)).item()

    print('Test Accuracy of the model on the {} test samples: {}'.format(len(testing_set_inputs), loss / len(testing_set_inputs)))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')