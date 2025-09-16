# SAR RARP50 Surgical Tool Segmentation

## Installation

This code is functional on `python>=3.8`, I have not tested other Python versions.

1. Clone the repository.
      ```
      git clone https://github.com/ZohraRezgui/segment.git
      cd segment
      ```

2. Create a virtual environment and activate the environment.
    ```
    conda create -n machnet python=3.8 -y
    conda activate machnet
    ```
3. Install  dependencies.
    ```
    pip install -r requirements.txt
    ```

## Data preparation

Once you download the data and you unzip it and rename the two folders to `train` and `test` accordingly. Then run the data restructuring code here to obtain the directory tree described in the README.md instructions of the dataset:

```bash
cd segment/utils/
python restructure.py --root /path/to/downloaded_data --outdir path/to/save/data
```

Then run these commands to unzip the files quickly to get the videos and remove the unnecessary zip files to save space:
``` bash
cd path/to/save/data
find . -name "*.zip" -exec sh -c 'unzip "$1" -d "${1%%.zip}"' _ {} \;
find . -name "*.zip" -delete
```

Now you have the data structure required to follow the preprocessing made available by the authors of the dataset to unpack the RGB video frames, please go ahead and follow the instructions provided [here](https://github.com/surgical-vision/SAR_RARP50-evaluation) using the `unpack` command.
I extracted 1fps from the videos instead of 10, this is easily done with the code provided by the dataset authors by just setting the parameter `-f` to 1:


``` bash
docker container run --rm \
                     -v /path_to_root_data_dir/:/data/ \
                     sarrarp_tk \
                     unpack /data/ -j4 -r -f 1
```


You should have the following structure eventually:

``` bash
data_root/
 ├── train1/
 │    ├── video_02/
 │    │     ├── rgb/
 │    │     ├── segmentation/
 │    │     ├── action_continuous.txt
 │    │     ├── action_discrete.txt
 │    │     ├── video_left.avi
 │    ├── ...
 ├── train2/
 │    ├── video_01/
 │    │     ├── rgb/
 │    │     ├── segmentation/
 │    │     ├── action_continuous.txt
 │    │     ├── action_discrete.txt
 │    │     ├── video_left.avi
 │    ├── ...
 ├── test/
 │    ├── video_45/
 │    │     ├── rgb/
 │    │     ├── segmentation/
 │    │     ├── action_continuous.txt
 │    │     ├── action_discrete.txt
 │    │     ├── video_left.avi
 │    ├── ...
```


## Training (Finetuning)

In this repo I only finetune a few layers of `DeepLabv3 + MobileNetV3 backbone` for segmentation that is available in the PyTorch library, due to it being lightweight and me having limited computational resources.
All parameters are defined in `config.py` file for training and evaluation.


- `config.combined_loss = "focal" `:  this is the loss that you want to use in combination with a Dice Loss, "ce" uses cross entropy, "focal" uses focal loss
- `config.unfreeze_last_block = True`  : The header classifier is always finetuned, but here you may finetune earlier layers, this unfreezes the last two blocks in the backbone and allows their parameters to be finetuned
- ` config.use_augmentation = True`  : This gives you the option to augment the training images with random cropping and flipping



After setting parameters in `config.py`, you run training as follows:

```bash
python train.py
```

## Evaluation

To calculate `IoU` per surgical tool, video and the overall `Iou`. You have to set in `config.py` which model you want to evaluate in `config.save_pth` and where you want the results to be saved in `config.res_dir` then run:



```bash
python evaluate.py
```

*Outputs:*

- Per-class IoU

- Per-video mean IoU

- Overall average IoU across videos

The class 0 corresponds to the background and is not taken into account in these calculations to not inflate the numbers.

## Inference on image or folder of images
Models are in checkpoint folder. Further below you may find the performance differences between models.
Run inference on a single image:



```bash
python inference.py \
  --model ./checkpoints/model_best.pth \
  --image ./sample.png \
  --output ./preds_single/
```


Run inference on a folder (video):

```bash
python inference.py \
  --model ./checkpoints/model_epoch20.pth \
  --folder ./data/test/video_45/rgb \
  --output ./preds/
```
## Finetuned Models Details
The Mean IoU corresponds to the mean of all IoU per videos in the test set.

| Model        | Training Strategy                                                                 | Mean IoU |
|--------------|-----------------------------------------------------------------------------------|----------|
| `model_1.pth` | Only classifier head fine-tuned with **Cross Entropy Loss**                      |  0.3030        |
| `model_2.pth` | Only classifier head fine-tuned with **Cross Entropy + Dice Loss**               |     0.3271     |
| `model_3.pth` | Classifier head + preceding conv layer fine-tuned with **Cross Entropy + Dice**  |   0.3202       |
| `model_4.pth` | Only classifier head fine-tuned with **Focal Loss + Dice Loss**                  |0.3298          |
| `model_5.pth` | Classifier head + preceding conv layer fine-tuned with **Focal + Dice**, with **Data Augmentation** |  0.3173        |
| `model_6.pth` | Entire last **Inverted Residual Block** fine-tuned with **Focal + Dice**, with **Data Augmentation** | 0.3913

## Visualization
After inference you can overlay the masks to the images or create a gif by using `viz.py` as follows, you can optionally set transparency for the overlay and frame duration:

-  an overlay gif example:

    ```bash
    python viz.py \
    --mode gif \
    --image_folder ./data/test/video_45/rgb \
    --mask_folder ./preds/video_45 \
    --output_path ./assets/overlay_pred.gif
    ```


-  visualizing the masks:

    ```bash
    python viz.py \
    --mode mask \
    --mask_folder ./preds/video_45 \
    --output_path ./assets/colored_masks/
     ```

## Demo
The following are frames from video_45:
Left is ground truth; right is the prediction with the `model_6.pth` model.



<p float="left">
    <img src="./assets/overlay_gt.gif" alt="Ground Truth Overlay" width="45%"/>
    <img src="./assets/overlay.gif" alt="Prediction Overlay" width="45%"/>
</p>
