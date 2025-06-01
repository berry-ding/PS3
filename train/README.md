# Pre-Training PS3

This codebase is built largely on top of [OpenCLIP](https://github.com/mlfoundations/open_clip). The main changes include **1)** adding the snippet to build PS3 model in `src/open_clip/model.py`, **2)** adding the image-text-box dataset in `src/open_clip_train/data.py`, **3)** adding the PS3 pre-training loss in `src/open_clip/loss.py`, and **4)** slightly modifying `src/open_clip_train/train.py` to support PS3 training. Common practices of training with OpenCLIP should be inherited in general.

## Installation

First make sure to install PS3 as instructed [here](https://github.com/NVLabs/PS3).

Then install this codebase as following (there's also instructions in [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/7260a46e7b4bcf518f5200fea06da5bc85aae025?tab=readme-ov-file#development)):

```bash
make install
make install-training
```

## Data Preparation

Trainin data should be in webdataset format, following the original OpenCLIP. Specifically, we use two separate webdatasets for images and text-box pairs. The image webdataset should be in the following structure:

```
images_path/
|-- image_00000.tar
|   |-- 00000001.jpg
|   |-- 00000002.jpg
|   |-- ...
|-- image_00001.tar
|   |-- 00010001.jpg
|   |-- 00010002.jpg
|   |-- ...
|-- ...
```

The text-box webdataset should have the exact same structure as the image webdataset, but with the text-box annotations in the format of `json` instead of images:

```
text_boxes_path/
|-- text_box_00000.tar
|   |-- 00000001.json
|   |-- 00000002.json
|   |-- ...
|-- text_box_00001.tar
|   |-- 00010001.json
|   |-- 00010002.json
|   |-- ...
|-- ...
```

Each json file contains pairs of local captions and local bounding boxes, as well as a global caption of the image. For example:

```json
{
  "text": [
    "The second image is a cropped view of the Ferris wheel and the clock tower from the first image. The Ferris wheel is prominently visible on the left side, with a green and white color scheme. The clock tower, featuring a white clock face and a green roof, is situated in the middle of the image. The sky appears overcast, and there are no other significant objects or text in the frame.",
    "The second image is a cropped view of the first image, focusing on the water and the distant cityscape. The image shows a body of water with a few boats and a dock in the foreground. In the background, there are several buildings and structures, including a large Ferris wheel and a tall skyscraper. The sky is clear with some clouds.",
    "The second image shows a cropped view of the first image, focusing on a section of the cityscape. The buildings are tall and modern, with various signs and advertisements visible. The text on the signs includes \"UBS,\" \"AXA,\" and \"PRUDENTIAL.\" The sky is clear, and the overall color palette is dominated by blues and grays, typical of an urban environment.",
    "The second image is a close-up crop of the first image, focusing on a section of the cityscape. It features tall skyscrapers with reflective glass facades, a few smaller buildings, and a construction site with scaffolding and a crane. There are also some trees and bushes in the foreground, and a few streetlights are visible. The sky is clear with a few clouds."
  ],
  "box": [
    [
      1020,
      780,
      1380,
      1020
    ],
    [
      2010,
      675,
      2190,
      1125
    ],
    [
      3075,
      810,
      3525,
      990
    ],
    [
      510,
      675,
      690,
      1125
    ]
  ],
  "global_text": "The image is a panoramic view of a cityscape with a prominent waterfront. On the left side, there is a tall skyscraper with a distinctive Ferris wheel in front of it. The Ferris wheel is surrounded by several other high-rise buildings. The middle ground features a large body of water, possibly a bay or harbor, with a few boats visible on the water. On the right side, there are more high-rise buildings, including a particularly tall skyscraper that stands out due to its height and design. The sky is clear with a few scattered clouds, suggesting a sunny day. The overall scene is vibrant and bustling, indicative of a major urban area."
}
```

Release of pre-training data is still under review. You can also build you own data following the pipeline described in the [paper](https://arxiv.org/abs/2503.19903).


## Training

Example trianing scripts for PS3-1.5K-SigLIP and PS3-4K-SigLIP are provided in `train/scripts`. The scripts are using single GPU node but we suggest using multiple nodes. The models in the paper are trained with 16 nodes of 8xA100.

## Convert the checkpoint to use with PS3 package

Since the checkpoint format is different from what is used by the [PS3 package](https://github.com/NVLabs/PS3) (which is huggingface format), we need to convert the checkpoint format with:

```bash
python -m src.open_clip.save_ps3_hf_ckpt --model <model_name> --pretrained <pretrained_path> --save-dir <save_dir>
```

After this you will be able to load the chkeckpoint with PS3 package, for example:

```bash
vision_model = PS3VisionModel.from_pretrained("path_after_conversion")
```

## More Information

We suggest checking out the original [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/7260a46e7b4bcf518f5200fea06da5bc85aae025?tab=readme-ov-file#training-clip) repository for other important tips on training, such as training with multiple data sources and more efficient training.


## Citing

If you found this repository useful, please consider citing PS3 and the original OpenCLIP repository:

```bibtex
@article{shi2025scaling,
  title={Scaling Vision Pre-Training to 4K Resolution},
  author={Shi, Baifeng and Li, Boyi and Cai, Han and Lu, Yao and Liu, Sifei and Pavone, Marco and Kautz, Jan and Han, Song and Darrell, Trevor and Molchanov, Pavlo and others},
  journal={arXiv preprint arXiv:2503.19903},
  year={2025}
}
```

```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

```bibtex
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```
