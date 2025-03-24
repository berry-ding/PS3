# Scaling Vision Pre-Training to 4K Resolution

[![website](https://img.shields.io/badge/website-76b900?style=for-the-badge&logo=safari&labelColor=555555)](https://nvlabs.github.io/PS3)
[![Arxiv](https://img.shields.io/badge/Arxiv-b31b1b?style=for-the-badge&logo=arxiv&labelColor=555555)]()
[![PS3 Models](https://img.shields.io/badge/PS3%20Models-ffd21e?style=for-the-badge&logo=huggingface&labelColor=555555)]()
[![VILA-HD Models](https://img.shields.io/badge/VILA--HD%20Models-ffd21e?style=for-the-badge&logo=huggingface&labelColor=555555)]()
[![VILA-HD Code](https://img.shields.io/badge/VILA--HD%20Code-181717?style=for-the-badge&logo=github&labelColor=555555)]()

<hr style="border: 2px solid gray;"></hr>

## Latest Updates
- [2025] Initial release.


<hr style="border: 2px solid gray;"></hr>

## Installation

Install through pip to use PS3 out of the box.
```bash
pip install ps3
```

If you would like to make changes to the PS3 code, clone this repo and install in editable mode.
```bash
cd PS3
pip install -e .
```

<hr style="border: 2px solid gray;"></hr>


## Quick Start

### Load Model and Image
```python
from PIL import Image
from ps3 import PS3VisionModel, PS3ImageProcessor

# Load the PS3 model and processor.
vision_model = PS3VisionModel.from_pretrained("NVlabs/PS3-4k-vit-b-16")
processor = PS3ImageProcessor.from_pretrained("NVlabs/PS3-4k-vit-b-16")
vision_model.cuda().eval()

# Create a dummy 4K image. Replace it with any of your own image.
image = Image.new("RGB", (3780, 3780), color=(256, 0, 0))

# Preprocess the image.
x = processor(image)["pixel_values"][0].unsqueeze(0).cuda()
```

### Encode High-Res Image with Bottom-Up Selection

PS3 can select important high-res patches baed on visual saliency and encode those patches.

**You can encode the whole high-res image using PS3.**
```python
out = vision_model(x, num_look_close="all").last_hidden_state
print(out.shape)  # (1, 88209, 1152)
```
Note the PS3-4K model processes the image at multiple scales: 378 (low-res), 756, 1512, and 3780, and it has a patch size of 14.

Then the number of tokens at each scale is (378/14)^2 = 729, (756/14)^2 = 2916, (1512/14)^2 = 11664, and (3780/14)^2 = 72900.

The output hidden state concatenates all the tokens along sequence dimension.
That gives us 729 + 2916 + 11664 + 72900 = 88209 tokens in total.

**You can encode parts of the high-res image by setting `num_look_close`, i.e., how many times to run the high-res selection and encoding.**
```python
out = vision_model(x, num_look_close=2).last_hidden_state
print(out.shape)  # (1, 5849, 1152)
```
In this example, it only runs the high-res selection and encoding for twice.

Note that PS3 processes at most 2560 high-res patches at a time. Then running high-res selection and encoding for twice gives us 2560 * 2 = 5120 high-res tokens. There is also 729 low-res tokens. That gives us 729 + 5120 = 5849 tokens in total.

**You can also decide how many high-res tokens to process by setting `num_token_look_close`.**
```python
out = vision_model(x, num_token_look_close=3000).last_hidden_state
print(out.shape)  # (1, 3729, 1152)
```
In this example, it only processes 3000 high-res tokens. Note that PS3 only processes 2560 high-res patches at a time. This means it needs to run the high-res selection and encoding for twice, with the first time processing 2560 high-res tokens and the second time processing 440 tokens. In the end it outputs 3729 tokens (3000 high-res + 729 low-res).


### Encode High-Res Image with Top-Down Selection

PS3 can also select important high-res patches based on any text prompt.

First of all, load the text model and encode the text prompt.
```python
from ps3 import PS3Tokenizer, PS3TextModel

tokenizer = PS3Tokenizer.from_pretrained("NVlabs/PS3-4k-vit-b-16")
text_model = PS3TextModel.from_pretrained("NVlabs/PS3-4k-vit-b-16")
text_model.cuda().eval()

text = ["a photo of a cat"]
text = tokenizer(text).cuda()
prompt = text_model(text).prompt
```

Then encode the image using text prompt to select high-res regions.
```python
out = vision_model(x, num_look_close=2, prompt=prompt).last_hidden_state
print(out.shape)  # (1, 5849, 1152)
```




<hr style="border: 2px solid gray;"></hr>

## Pre-Trained Models


<hr style="border: 2px solid gray;"></hr>

## Inference

[Quick Start](#quick-start) gives some examples of how to use PS3 to encode an image. Below are more detailed explanations of the arguments of model inference.

```python
class PS3VisionModel(PS3PreTrainedModel):
    ...
    def forward(
        self,
        pixel_values, 
        num_look_close, 
        num_token_look_close=None, 
        prompt=None, 
        gt_selection_maps=None, 
        smooth_selection_prob=False,
        only_select_first_n_scale=None,
        is_global_text=None, 
        pool_gt_token_only=False, 
    ):
    ...
```
`pixel_values`: the input images with shape (B, C, H, W).

`num_look_close`: how many times to run high-res selection and encoding. PS3 selects and processes 2560 patches each time. If set to `all` then it selects all the high-res patches. If set to `0` then PS3 only returns the low-res features. If set to a larger number than what it needs to encode all the high-res patches, then PS3 will clamp it to the max number needed.

`num_token_look_close`: (optinoal) how many high-res patches to select and process. Similar to `num_look_close` but counts the number of high-res tokens instead of number of running high-res encoding.

`prompt`: (optional) the prompt embedding used to select high-res patches. The prompt embedding can be embedding of some text, or some embedding output by an LLM (see paper). The shape of prompt embedding is (B, C) where B is the batch size (same in `pixel_values`) and C is the embedding dimension (same as PS3 token embedding dimension). If `prompt=None`, then PS3 will select high-res patches based on visual saliency (bottom-up selection).

`gt_selection_maps`: (optional) the ground truth selection maps for the image. It should be a tensor of 0/1 values with shape (B, h, w). Regions with value 1 means they should be selected. When selectin high-res patches, PS3 will interpolate the `gt_selection_maps` to the same size as the feature map at each scale, prioritize selecting the tokens where the value is 1, and if there's still budget for selecting more tokens, select the rest based on the original selection probability.

`smooth_selection_prob`: (optional) smooth the selectino probability map such that the selected patches won't be distributed too scarcely each time it runs high-res selection. It slightly improves the performance occasinoally when selecting all the patches but usually hurts when selecting parts of the patches.

`only_select_first_n_scale`: (optional) only select the first n high-res scales. For example, for PS3-4K model, if `only_select_first_n_scale=2`, then only select and process scales of 756 and 1512, and ignore the scale of 3780.

`is_global_text`: (optional) only return the pooled low-res feautres. *It will only be used during pre-training.*

`pool_gt_token_only`: (optional) only pool the tokens inside the gt selection regions. *It will only be used during pre-training.*




