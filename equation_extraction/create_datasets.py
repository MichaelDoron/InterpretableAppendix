import time
import pandas as pd
import torch
import torchvision
from PIL import Image
from skimage import io, transform

data = pd.read_csv('ball_data.csv')

world_nums = np.arange(0, data['world_nums'].max() + 1)
steps = np.arange(0, data['steps'].max() + 1)

number_of_samples = 10000
number_of_previous_frames = 10

world_indices = np.random.choice(world_nums, number_of_samples)
step_indices = np.random.randint(0, len(steps) - number_of_previous_frames, size=(number_of_samples))
	
X = torch.Tensor(0,3,60,80)
Y = []
T = []
start  = time.time()
for sample in range(number_of_samples):
	t = np.random.randint(4,9)
	world_ind = world_indices[sample]
	step_ind = step_indices[sample]
	if (len(data.index[(data['world_nums'] == world_ind) & (data['steps'] == (step_ind + 9))]) == 0): continue # check if sample exists
	line = data.index[(data['world_nums'] == world_ind) & (data['steps'] == step_ind)][0]
	if (len(data.index[(data['world_nums'] == world_ind) & (data['steps'] == step_ind)]) == 0): continue # check if sample exists
	if data.iloc[line + 9]['world_nums'] != world_ind: continue # check if there are 9 more frames
	frame_1 = torchvision.transforms.functional.to_tensor(torchvision.transforms.functional.to_grayscale(Image.open(data.iloc[line + 0]['frame_img']).resize((60, 80))))
	frame_2 = torchvision.transforms.functional.to_tensor(torchvision.transforms.functional.to_grayscale(Image.open(data.iloc[line + 1]['frame_img']).resize((60, 80))))
	sample_images = torch.cat((frame_1, frame_2), 0)
	sample_images = sample_images[None, :, :, :]
	X = torch.cat((X, sample_images), 0)
	Y.append((data.iloc[line + t]['X'], data.iloc[line + t]['Y']))
	T.append(t)
	print('Done sample {} in {}'.format(sample, time.time() - start))
	start = time.time()

Y = torch.Tensor(Y)

torch.save(X, "inputs_X.pt")
torch.save(T, "inputs_t.pt")
torch.save(Y, "outputs.pt")