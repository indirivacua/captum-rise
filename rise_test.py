# %%
from rise import RISE

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import os
import json
from time import perf_counter

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_printoptions(precision=4, sci_mode=False)

DEVICE, DTYPE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.float32,
)
MODEL_NAME = "resnet50"
INPUT_PATH = "img/resnet/"
OUTPUT_PATH = "outputs"

labels_path = "models/imagenet_class_index.json"

os.makedirs(OUTPUT_PATH, exist_ok=True)

model = getattr(models, MODEL_NAME)(weights="DEFAULT").to(device=DEVICE, dtype=DTYPE)
model = nn.Sequential(
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    model,
    nn.Softmax(dim=1),
)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device=DEVICE, dtype=DTYPE)),
    ]
)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Model {type(model).__name__} total parameters: ", pytorch_total_params)

with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)

get_class = lambda id: idx_to_labels[str(id)][0]
get_label = lambda id: idx_to_labels[str(id)][1]

load_image = lambda img_source, transform: transform(Image.open(img_source))

inputs = []
for filename in os.listdir(INPUT_PATH):
    f = os.path.join(INPUT_PATH, filename)
    image = load_image(f, transform)
    inputs.append(image)
inputs = torch.stack(inputs, dim=0)

# %%
with torch.no_grad():
    output = model(inputs)
prediction_score, pred_label_idx = map(lambda x: x.squeeze_(), torch.topk(output, 1))

for score, label_idx in zip(prediction_score, pred_label_idx):
    id = label_idx.item()
    predicted_label = get_label(id)
    print(f"Predicted: {predicted_label} ({id}) ({score.item():.4f})")

# %%
rise = RISE(model)

n_masks_list = [
    2**7,
]  # 2**8, 2**9, 2**10, 2**11, 2**12, 2**13]
initial_mask_shape_list = [
    (2, 2),
]  # (4, 4), (8, 8), (16, 16)]

for n_masks in n_masks_list:
    for initial_mask_shape in initial_mask_shape_list:
        print(n_masks, initial_mask_shape)
        start_time = perf_counter()
        heatmap = rise.attribute(
            inputs,
            n_masks=n_masks,
            initial_mask_shapes=(initial_mask_shape,),
            target=pred_label_idx,
            show_progress=True,
        )
        heatmap = heatmap.squeeze().cpu()

        print("Elapsed time: ", perf_counter() - start_time)
        print("Heatmap shape: ", heatmap.shape)
        print("Min and max heatmap values: ", heatmap.min(), heatmap.max())

        fig, axs = plt.subplots(inputs.shape[0], 3, constrained_layout=True)

        fig.set_size_inches(5, inputs.shape[0])
        fig.set_dpi(200)

        pc = [None] * inputs.shape[0]
        for idx_in, input in enumerate(inputs):
            input = np.moveaxis(input.squeeze().cpu().numpy(), 0, -1)
            axs[idx_in, 0].imshow(input)
            # plt.axis("off")
            pc[idx_in] = axs[idx_in, 1].imshow(
                heatmap[idx_in, :, :].numpy(), cmap="jet"
            )
            # plt.axis("off")
            axs[idx_in, 2].imshow(input)
            axs[idx_in, 2].imshow(heatmap[idx_in, :, :].numpy(), cmap="jet", alpha=0.5)
            # plt.axis("off")
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        for idx_in in range(inputs.shape[0]):
            fig.colorbar(
                pc[idx_in],
                ax=axs[idx_in, :],
            )  # shrink=0.75)
        plt.savefig(
            f"{OUTPUT_PATH}/rise{idx_in}_masks{n_masks}_ishape{initial_mask_shape}.png"
        )
        plt.close()
