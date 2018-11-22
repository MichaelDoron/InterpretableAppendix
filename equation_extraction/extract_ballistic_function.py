import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import argparse
import math
import pickle

parser = argparse.ArgumentParser(description='extract_ballistic_function')
parser.add_argument(
    '--train_model',
    action='store_true', 
    default=False,    
    help='Whether to train the model')
parser.add_argument(
    '--train_appendix',
    action='store_true', 
    default=False,
    help='Whether to train the appendix')
parser.add_argument(
    '--constraint_sparsity',
    action='store_true', 
    default=False,
    help='Whether to constaint the original network')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of ep')

def create_trajectories():
    ress = []
    for t in range(20):
        pred_y = appendix(x[:,:,:,:], torch.Tensor([t] * len(testing_set_outputs)))
        ress.append(pred_y.detach().numpy())

    ress = np.array(ress)
    plt.plot(ress)

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
        self.fc1 = nn.Linear(9601, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)        
        
    def forward(self, x, t):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = torch.cat((out.reshape(out.size(0), -1), t.reshape(t.size(0), -1)), dim=1)
        out = torch.nn.functional.relu(self.fc1(out))
        out = torch.nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class Ballistic(torch.nn.Module):
    def __init__(self):
        super(Ballistic, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(6)) 
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)      

    def forward(self, input_1, t):
      # Want to learn this: y_0           + V_0y          * 10     * t_coeff        - (g / 2)        *          (10     * t_coeff)        ^ 2 
      output_x =              input_1[:, 0] * self.weight[0] + input_1[:, 1] * t * self.weight[1]
      output_y =              input_1[:, 2] * self.weight[2] + input_1[:, 3] * t * self.weight[3] + input_1[:, 4] * self.weight[4] * torch.pow(t * input_1[:, 5] * self.weight[5], 2)
      return torch.cat((output_x.reshape(-1,1), output_y.reshape(-1,1)), dim=1)

def appendix_forward(self, x, t):
    app_tensor = torch.Tensor()
    out = self.cnn1(x)
    out = self.cnn2(out)
    out = torch.cat((out.reshape(out.size(0), -1), t.reshape(t.size(0), -1)), dim=1)
    out = torch.nn.functional.relu(self.fc1(out))
    app_tensor = torch.cat((app_tensor, out.clone()), 1)
    out = torch.nn.functional.relu(self.fc2(out))
    app_tensor = torch.cat((app_tensor, out.clone()), 1)
    app_out = self.app_fc(app_tensor)
    app_out = self.app_Ballstic(app_out, t)
    return app_out        

def train_model():
    print('training model')
    model = ConvNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    original_losses = []
    epoch_losses = []
    # Train the model
    for epoch in range(num_epochs):
        if np.mean(epoch_losses) < 50:
            break
        epoch_losses = []
        for (images, ts, positions) in batches:
            x = images.to(device)
            t = ts.to(device)
            y = positions.to(device)
            
            pred_y = model(images, t)
            loss = torch.sqrt(criterion(pred_y, y))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            original_losses.append(loss.item())
        print ('Epoch [{}/{}], Mean loss: {:.4f}' .format(epoch+1, num_epochs, np.mean(epoch_losses)))
        torch.save(model.state_dict(), 'model.ckpt')

    # Test the model
    model.eval()
    with torch.no_grad():
        average_testing_loss = 0
        x = testing_set_inputs_X.to(device)
        t = testing_set_inputs_t.to(device)
        y = testing_set_outputs.to(device)
        pred_y = model(x, t)
        loss = torch.sqrt(criterion(pred_y, y)).item()

        print('Test Accuracy of the model on the {} test samples: {}'.format(len(testing_set_inputs_X), loss))

    # Save the model checkpoint
    pickle.dump(original_losses, open('original_losses.pickle','wb'))

def train_appendix():
    print('training appendix')
    model = ConvNet().to(device)
    state_dict = torch.load('model.ckpt')
    model.load_state_dict(state_dict)
    appendix = ConvNet()
    appendix.load_state_dict(model.state_dict())
    appendix.forward = appendix_forward.__get__(appendix)
    appendix.app_fc = nn.Linear(64, 6)        
    appendix.app_Ballstic = Ballistic()
    for name, param in appendix.named_parameters():
      if name.startswith('app_'):
        param.requires_grad = True
      else:
        param.requires_grad = False
      print ('{}.requires_grad = {}'.format(name, param.requires_grad))
    appendix_losses = []
    epoch_losses = []
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(appendix.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        if np.mean(epoch_losses) < 50:
            break
        epoch_losses = []
        for (images, ts, positions) in batches:
            t = ts.to(device)
            x = images.to(device)
            y = positions.to(device)
            pred_y = appendix(images, t)
            loss = torch.sqrt(criterion(pred_y, y))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            appendix_losses.append(loss.item())
            epoch_losses.append(loss.item())
        torch.save(appendix.state_dict(), 'appendix.ckpt')
        print ('Epoch [{}/{}], Mean loss: {:.4f}' .format(epoch+1, num_epochs, np.mean(epoch_losses)))

    # Test the appendix
    appendix.eval()
    with torch.no_grad():
        average_testing_loss = 0
        testing_set_inputs_X = inputs_X[testing_indices,:,:,:]
        testing_set_inputs_t = torch.Tensor(inputs_t)[testing_indices]
        testing_set_outputs = outputs[testing_indices,:]

        x = testing_set_inputs_X.to(device)
        t = testing_set_inputs_t.to(device)
        y = testing_set_outputs.to(device)
        pred_y = appendix(x, t)
        loss = torch.sqrt(criterion(pred_y, y)).item()

        print('Test Accuracy of the appendix on the {} test samples: {}'.format(len(testing_set_inputs_X), loss))
    pickle.dump(appendix_losses, open('appendix_losses.pickle','wb'))

def constraint_sparsity():
    appendix = ConvNet().to(device)
    appendix.forward = appendix_forward.__get__(appendix)
    appendix.app_fc = nn.Linear(64, 6)        
    appendix.app_Ballstic = Ballistic()
    state_dict = torch.load('appendix.ckpt')
    appendix.load_state_dict(state_dict)

    for name, param in appendix.named_parameters():
        param.requires_grad = True
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(appendix.parameters(), lr=learning_rate)

    l1_weight = 0
    l1_loss = 0
    epoch_loss = [np.inf]
    sparse_appendix_losses = []
    sparsity_losses = []
    for epoch in range(num_epochs):
        for (images, ts, positions) in batches:
            t = ts.to(device)
            x = images.to(device)
            y = positions.to(device)
            pred_y = appendix(images, t)
            optimizer.zero_grad()
            to_regularise = []
            for name, param in appendix.named_parameters():
                if 'app_Ballstic' not in name:
                    to_regularise.append(param.view(-1))
            l1_loss = l1_weight * torch.sum(torch.abs(torch.cat(to_regularise)))
            loss = torch.sqrt(criterion(pred_y, y)) + l1_loss
            sparse_appendix_losses.append(torch.sqrt(criterion(pred_y, y)).item())
            sparsity_losses.append(l1_loss.item())
            epoch_loss.append(torch.sqrt(criterion(pred_y, y)).item())
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], mean loss: {:.4f}, sparsity weight = {:.4f}' .format(epoch+1, num_epochs, np.mean(epoch_loss), l1_weight))
        if np.mean(epoch_loss) < 100:
            l1_weight += 0.05
        epoch_loss = []
        torch.save(appendix.state_dict(), 'sparse.ckpt')
    pickle.dump(sparsity_losses, open('sparsity_losses.pickle','wb'))
    pickle.dump(sparse_appendix_losses, open('sparse_appendix_losses.pickle','wb'))


if __name__ == '__main__':
    args = parser.parse_args()
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = args.epochs
    batch_size = 50
    learning_rate = 0.001

    inputs_X = torch.load('inputs_X.pt')
    inputs_t = torch.load('inputs_t.pt')
    outputs = torch.load('outputs.pt') * 1000

    training_set_percentage = 0.5
    testing_set_percentage = 1 - training_set_percentage

    training_indices = np.random.choice(np.arange(len(inputs_t)), int(len(inputs_t) * training_set_percentage), replace = False)
    testing_indices = np.array(list(set(range(len(inputs_t))) - set(training_indices)))

    training_set_inputs_X = inputs_X[training_indices,:,:,:]
    training_set_inputs_t = torch.Tensor(inputs_t)[training_indices]
    training_set_outputs = outputs[training_indices,:]

    testing_set_inputs_X = inputs_X[testing_indices,:,:,:]
    testing_set_inputs_t = torch.Tensor(inputs_t)[testing_indices]
    testing_set_outputs = outputs[testing_indices,:]

    batches = []
    total_training_indices = np.arange(len(training_set_inputs_X))
    for batch in range((len(training_set_inputs_X) / batch_size) + 1):
        if len(total_training_indices) >= batch_size:
            batch_indices = np.random.choice(total_training_indices, batch_size, replace = False)
        else:
            batch_indices = total_training_indices
        total_training_indices = np.array(list(set(total_training_indices) - set(batch_indices)))
        batches.append((training_set_inputs_X[batch_indices, :, :, :], training_set_inputs_t[batch_indices], training_set_outputs[batch_indices, :]))

    if args.train_model: train_model()
    if args.train_appendix: train_appendix()
    if args.constraint_sparsity: constraint_sparsity()




# t = np.arange(0,20,1)
# X0 = [-1] * 20
# Y0 = [-1] * 20
# X_vel = [1] * 20
# Y_vel = [-5] * 20
# output_x = X0 + X_vel * t * (-2.1311)
# output_y = Y0 + Y_vel * t * (2.4125) + (1.4394) * np.power(t * (1.5180), 2)
# plt.close('all')
# plt.plot(output_y)

# for name, param in appendix.named_parameters():
#     if 'app_' in name:
#         print(name, param)


# for (images, ts, positions) in batches:
#     pred_y = []
#     for t in [4,5,6,7,8]:
#         ts = torch.Tensor([t] * len(ts)).to(device)
#         x = images.to(device)
#         y = positions.to(device)
#         pred = appendix(images, ts)
#         pred_y.append(pred[:,1].detach().numpy())

