import torch
import numpy as np

device = torch.device('cpu')
# device = torch.device('cuda')

number_of_coeffs = 10

coeffs = np.random.rand(number_of_coeffs)

def poly_function(x):
  result = 0
  for ind in range(len(coeffs)):
    result += torch.pow(x, ind) * coeffs[ind]
  return result

N, D_in, H, D_out = 64, 1, 20, 1

validation_x = torch.from_numpy(np.linspace(-1, 1, 400).reshape(-1,1)).float()
validation_y = poly_function(validation_x)

class polynomial(torch.nn.Module):
    def __init__(self, in_features):
        super(polynomial, self).__init__()
        self.in_features = in_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)      

    def forward(self, input_1):
      output_repeat = input_1.repeat(1, self.in_features)
      output_power  = torch.zeros(input_1.shape[0],1)
      for ind in range(self.in_features):
        output_power += (torch.pow(output_repeat[:, ind], ind) * self.weight[ind]).view(input_1.shape[0],1)
      return output_power


appendix = polynomial(number_of_coeffs)

for param in appendix.parameters():
  param.requires_grad = True

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4

training_losses = []
validation_losses = []

number_of_batches = 10
for batch in range(number_of_batches):
  if (batch > 0) and (loss.item() < 1e-1):
    break
  x = torch.randn(N, D_in, device=device)
  y = poly_function(x)
  for t in range(10000):
    y_pred = appendix(x)
    loss = loss_fn(y_pred, y)
    training_losses.append(loss.item())
    optimizer = torch.optim.Adam(appendix.parameters(), lr = learning_rate)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    validation_y_pred = appendix(validation_x)
    validation_loss = loss_fn(validation_y_pred, validation_y)
    validation_losses.append(validation_loss.item())
    print('batch {}, step {}, validation loss {}, loss {}'.format(batch, t, validation_loss.item(),loss.item()))
    


final_x = torch.from_numpy(np.linspace(-1, 1, 400).reshape(-1,1)).float()
final_y = poly_function(final_x)
final_y_pred = appendix(final_x)

plt.close('all')
plt.plot(final_x.numpy(), final_y.numpy())
plt.plot(final_x.numpy(), final_y_pred.detach().numpy())


