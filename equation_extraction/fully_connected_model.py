import torch
import numpy as np

device = torch.device('cpu')
# device = torch.device('cuda')

number_of_coeffs = 5

coeffs = np.random.rand(number_of_coeffs)

def poly_function(x):
  result = 0
  for ind in range(len(coeffs)):
    result += torch.pow(x, ind) * coeffs[ind]
  return result

N, D_in, H, D_out = 64, 1, 20, 1

validation_x = torch.from_numpy(np.linspace(-1, 1, 400).reshape(-1,1)).float()
validation_y = poly_function(validation_x)

model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        ).to(device)


appendix = model

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


