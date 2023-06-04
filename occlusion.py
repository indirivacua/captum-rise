# %%
import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

# %%
model = models.resnet18(weights='IMAGENET1K_V1') #pretrained=True
model = model.eval()

# %%
labels_path = 'models/imagenet_class_index.json' #os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
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

img = Image.open('img/resnet/goose.jpg')

transformed_img = transform(img)

input = transform_normalize(transformed_img)
input = input.unsqueeze(0)

# %%
output = model(input)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

pred_label_idx.squeeze_()
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

# %%
occlusion = Occlusion(model)

# attributions_occ = occlusion.attribute(input,
#                                        strides = (3, 8, 8),
#                                        target=pred_label_idx,
#                                        sliding_window_shapes=(3,15, 15),
#                                        baselines=0)

attributions_occ = occlusion.attribute(input,
                                       strides = (3, 50, 50),
                                       target=pred_label_idx,
                                       sliding_window_shapes=(3,60, 60),
                                       baselines=0)

# %%
import matplotlib.pyplot as plt
from datetime import datetime

fig, axs = plt.subplots(1,3, constrained_layout=True)
pc = [None] * 1

transformed_img_np = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0))
heatmap = np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0))
axs[0].imshow(transformed_img_np)
#plt.axis("off")
pc[0] = axs[1].imshow(heatmap[:,:,0], cmap='jet')
#plt.axis("off")
axs[2].imshow(transformed_img_np)
axs[2].imshow(heatmap[:,:,0], cmap='jet', alpha=0.5)
#plt.axis("off")
for ax in axs.flat:
  ax.set_xticks([])
  ax.set_yticks([])
fig.colorbar(pc[0], ax=axs[2], shrink=0.39)

now = datetime.now()
date_time = now.strftime("%d-%m-%Y-%H-%M-%S")

plt.savefig(f"results/occlusion_{date_time}.png", bbox_inches='tight')
plt.close()

# %%
