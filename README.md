# Experiments for Image Captioning using slot attention: 
This is my final year thesis project for improving performance on image captioning. 

## Current architecture: 
![ezgif-1-221093d2b5](https://user-images.githubusercontent.com/63205095/218675236-6e42396c-f94d-4dce-aded-e26f1f71da21.gif)

## Steps to run the code: 

### 1. Create a conda env: 

Create an environment: 
```
conda create --name imgcap
```
Activate the environment: 
```
conda activate imgcap
```
Install the dependancies: 
```
pip install -r requirements.txt
```

### 2. Downloading the data: 

Create a directory: `~/datasets/` using: 
```
mkdir ~/datasets/
```
#### Flickr8k: 

*Download the dataset:*
```
wget -P ~/datasets/gqa_imgs https://www.kaggle.com/datasets/adityajn105/flickr8k/download?datasetVersionNumber=1
```
*Extract the dataset: *
```
unzip flickr8k.zip
```

#### GQA: 

*Downloading GQA Images:*

Create a subdirectory: `~/datasets/gqa_imgs` using: 
```
mkdir ~/datasets/gqa_imgs
```
Download GQA imgs: 
```
wget -P ~/datasets/gqa_imgs https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
```

Extract the images folder: 
```
unzip images.zip
```

*Downloading GQA Annotations:*

Create a subdirectory: `~/datasets/gqa_ann` using: 
```
mkdir ~/datasets/gqa_ann
```
Download GQA annotations: 
```
wget -P ~/datasets/gqa_imgs https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1
```

Rename and extract annotations folder: 

```
mv 'mdetr_annotations.tar.gz?download=1' ann.tar.gz
tar -xvzf ann.tar.gz
```

### 3. Running the code on a single GPU system: 

Log into wandb 
```
wandb login
```
(This will prompt a link to the wandb auth key. Copy and paste it in the terminal.)

Pre-training on GQA dataset: 

```
python train.py --epochs=35 
```

Training on PhraseCut dataset: 

```
python train_obj.py --epochs=10 --load=1 
```
Fine-tuning on GQA dataset: 

```
python train.py --epochs=25 --load=1
```
### 3. Running the code on a multi-gpu system:

Get back on the home folder
``` 
cd slotvqa
```
Schedule the job: 
```
srun test.job.sbatch
```
