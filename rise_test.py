# %%

import torch
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image

import json

import os

from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


model = models.resnet50(weights='IMAGENET1K_V2')
model = model.eval()

model.fc = nn.Sequential(
    model.fc,
    nn.Softmax(dim=1),
)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

# %%
labels_path = 'models/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)

# %%
transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

imgs = []
for filename in os.listdir('img/resnet/'):
    f = os.path.join('img/resnet/', filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = Image.open(f)
        imgs.append(img)

inputs = []
transformed_imgs = []
for img in imgs:
    transformed_img = transform(img)

    input = transform_normalize(transformed_img)
    input = transformed_img
    input = input.unsqueeze(0)

    inputs.append(input)
    transformed_imgs.append(transformed_img)

inputs = torch.cat(inputs, dim=0)
inputs = (inputs,)

print(inputs[0].shape)

# %%
output = model(inputs[0]) #had to use [0] because the model only accepts one input at a time
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx = pred_label_idx[:,0]

for score, label_idx in zip(prediction_score, pred_label_idx):
  id =label_idx.item()
  predicted_label = idx_to_labels[str(id)][1]
  print(f"Predicted: {predicted_label} ({id}) ({score.squeeze().item()})")

# %%
# import sys
# sys.path.append('..')

# from rise import RISE

# %%

if torch.cuda.is_available():
  print("Using CUDA")

  device = torch.device("cuda")

  model.to(device)
  inputs = tuple(i.to(device) for i in inputs)

os.makedirs('results', exist_ok=True)

from rise import RISE
rise = RISE(model)

from time import perf_counter

for n_masks in [2**11,2**12,2**13]:#[2**7,2**10,2**12,2**13,2**14]:
    for initial_mask_shape in [(2,2),(4,4),(8,8),(16,16)]:#[(4,4), (7,7), (8,8), (15,15)]
        print(n_masks,initial_mask_shape)
        start_time = perf_counter()
        heatmap = rise.attribute(inputs, n_masks=n_masks, initial_mask_shapes=(initial_mask_shape,), target=pred_label_idx, show_progress=True).cpu()

        print("Elapsed time: ", perf_counter() - start_time)
        print("Heatmap shape: ", heatmap.shape)
        print("Min and max heatmap values: ", heatmap.min(),heatmap.max())



        fig, axs = plt.subplots(len(inputs[0]),3, constrained_layout=True)

        fig.set_size_inches(20,8)
        fig.set_dpi(500)

        pc = [None] * len(inputs[0])
        for idx_in, transformed_img in enumerate(transformed_imgs):
            transformed_img = np.moveaxis(transformed_img.squeeze().cpu().numpy(), 0, -1)
            axs[idx_in,0].imshow(transformed_img)
            #plt.axis("off")
            pc[idx_in] = axs[idx_in,1].imshow(heatmap[idx_in,:,:].numpy(), cmap='jet')
            #plt.axis("off")
            axs[idx_in,2].imshow(transformed_img)
            axs[idx_in,2].imshow(heatmap[idx_in,:,:].numpy(), cmap='jet', alpha=0.5)
            #plt.axis("off")
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        for idx_in in range(len(inputs[0])):
            fig.colorbar(pc[idx_in], ax=axs[idx_in, :], shrink=0.75)
        plt.savefig(f"results/rise{idx_in}_masks{n_masks}_ishape{initial_mask_shape}.png")
        plt.close()



# %%
