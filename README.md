# Captum RISE API Design

## Background

RISE (Randomized Input Sampling for Explanation) is a perturbation based approach to compute attribution. RISE uses a Monte-Carlo approximation algorithm to detect the sensitivity of an output value with respect to input features. RISE is a post-hoc, local and model-agnostic interpretability method that can be used any type of image-like classification network to explain its decisions. RISE generates a pixel importance map (_saliency_), indicating how salient is each pixel with respect to the model's predictions. It estimates the sensitivity of each input feature by sampling `n_masks` random occlusion masks, and computing the network output for each of the correspondingly occluded input images. Each mask is assigned a score based on the output of the model. Afterwards, masks are averaged, using the score as weight.

To sample occlusion masks, RISE assumes a strong spatial structure in the feature space, so that features that are close to each other are more likely to be correlated. It does not assume access to the parameters of the model, features or gradients and therefore does not require accessing the internals of the network architecture. When RISE is applied to images, it allows visualizing the regions of the image that were important for the prediction of a certain class.

Requires:

* $I$: input of size $H \times W$ in the space of  images $\mathcal{I} = \lbrace I | I : \Lambda \rightarrow \mathbb{R}^{d} \rbrace$ with $\Lambda = \lbrace 1,\dots,H \rbrace \times \lbrace 1,\dots,W \rbrace$.
* $M: \Lambda \rightarrow [0,1]$: $N$ random binary bilinear upsampled masks.
* $f: \mathcal{I} \rightarrow \mathbb{R}$: black-box model that for a given input produce a scalar confidence score (for example, the probability of a single class).

Mathematical formula:

$$
\begin{aligned}
S_{I,f}(\lambda) \stackrel{MC}{\approx} \frac{1}{\mathbb{E}[M] \cdot N} \sum_{i=1}^{N} f(I \odot M_{i}) \cdot M_{i}(\lambda)
\end{aligned}
$$

$S_{I,f}$ is the importance map, explaining the decision of model $f$ on input image $I$, normalized to the expectation of $M$. The importance of a pixel $\lambda \in \Lambda$, is the score over all possible masks $M$ conditioned on the event that pixel $\lambda$ is observed, i.e., $M(\lambda) = 1$. Finally, $f(I \odot M_{i})$ is a random variable that represents the model's output for the target class by performing the inference over the masked input, where $\odot$ denotes the Hadamard product (element-wise multiplication).

### Pseudocode:

```
# Algorithm 1: Mask Generation
  # Require: input_shape = (HxW), initial_mask_shape = (hxw)
  # Ensure: upsample_shape = (h+1)*CH x (w+1)*CW (where CHxCW = ceil(H/h) x ceil(W/w))
Repeat N times:
  Create a binary mask_i of size initial_mask_shape
  Convert Binary Mask to Smooth Mask:
    Upsample mask_i via bilinear interpolation to size upsampled_shape
    Crop upsampled mask_i randomly with a window of size input_shape
  Store the resulting mask_i inside the masks array or list
return masks
```

```
# Algorithm 2: Importance Map Generation
Repeat N times:
  masked_input i = mask input with mask_i
  Compute the model prediction for the masked input (weight_i of mask_i)
  Add to the heatmap the mask_i multiplied by its weight_i
Normalize the final heatmap by the number n_masks of masks generated
return heatmap
```

More details regarding the method can be found in the original [paper](https://arxiv.org/abs/1806.07421) and in the [RISE](https://github.com/eclique/RISE) implementation.

## Design Considerations:

* RISE works for generic inputs shapes provided by the user.
* The return value is the importance map.
* This implementation uses the extendes the Feature Ablation capabilities of Captum with the same API, making the code simple to mantain and learn.

<!--
## Proposed Captum API Design:

The Captum API includes a base class, which is completely generic and allows for implementations of generalized versions of surrogate interpretable model training.

The RISE implementation builds upon this generalized version with a design that closely mimics other attribution methods for users to easily try RISE and compare to existing attribution methods under certain assumptions on the function and interpretable model structure.
-->

## RISE

The RISE class is a generic class which allows training any interpretable model based on by sub-sampling the input image via random masks and recording its response to each of the masked images. The constructor takes only the model forward function. The RISE class makes certain assumptions to the generic SurrogateModelAttribution class in order to match the structure of other attribution methods and allow easier switching between methods.

The RISE class makes certain assumptions to the generic FeatureAblation class in order to match the structure of other attribution methods and allow easier switching between methods.

### **Constructor:**

```
RISE(forward_func: Callable)
```

**Argument Descriptions:**

* `forward_func` - `torch.nn.Module` corresponding to the forward function of the model or any modification of it, for which attributions are desired. This is consistent with all other attribution constructors in Captum.

### **attribute:**

```
attribute(input_set: TensorOrTupleOfTensorsGeneric,
          n_masks: int,
          initial_mask_shapes: TensorOrTupleOfTensorsGeneric,
          baselines: BaselineType = None,
          target: TargetType = None,
          additional_forward_args: Any = None,
          show_progress: bool = False,
          ) -> TensorOrTupleOfTensorsGeneric:
```

**Argument Descriptions:**

These arguments follow standard definitions of existing Captum methods. `inputs` is the input for which RISE attributions are computed. The type of it is `Tensor or tuple[Tensor, ...]` because if `forward_func` takes a single tensor as input, a single input tensor should be provided. And also, if `forward_func` takes multiple tensors as input, a tuple of the input tensors should be provided. It is assumed that for all given input tensors, dimension 0 corresponds to the number of examples (aka batch size), and if multiple input tensors are provided, the examples must be aligned appropriately. `initial_mask_shapes` is the initial mask shapes for each input of the model. The type of it is `Tensor or tuple[Tensor, ...]`. `target` is the target class for each input of the model.

A custom class named `MaskSetConfig` is in charge of computing the generation of the set of masks for the preconfigured input shapes, types and initial mask shapes. By default, assumes each input is of shape `[B,C,D1,D2,D3]`, where `B` and `C` are the batch and channel dimensions which are ignored, and the final mask sizes are extracted as `[D1,D2,D3]`. Dimensions `D2` and `D3` are optional. Because of pytorch's `interpolate` limitations, this only supports 5D inputs (3D masks) at most.
