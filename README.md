# Welcome to CBSI!
**Contrast-free BBB Status Identification model (CBSI)** is a generative diffusion AI that can identify BBB status with high accuracy using non-contrast MR images including T1 and T2-FLAIR MR scans.

<img src="https://github.com/SMU-MedicalVision/CBSI-master/blob/main/sample_images/Framework.png" width="800px">


This repository contains the code to our paper "Contrast-free identification of glioma blood-brain barrier status via generative diffusion AI and non-contrast MRI".



# Example
| Input               |Output 1              |Output 2                        |
|------------------------------|-----------------|-----------------|
|`T1 and T2-FLAIR MR scans` |`T1Gd MR scan` |`BBB status` |
|<img src="https://github.com/SMU-MedicalVision/CBSI-master/blob/main/sample_images/T1.png" width="90px"><img src="https://github.com/SMU-MedicalVision/CBSI-master/blob/main/sample_images/T2F.png" width="90px">|<img src="https://github.com/SMU-MedicalVision/CBSI-master/blob/main/sample_images/Synthetic_T1Gd.png" width="90px">| _Disrupted_|

# System Requirements
This code has been tested on Ubuntu in PyTorch and an NVIDIA GeForce RTX 3090 GPU. 

# Setup Environment
In order to run our model, we suggest you create a virtual environment
```
conda create -n CBSI_env python=3.8
```
and activate it with
```
conda activate CBSI_env
```
Subsequently, download and install the required libraries by running
```
pip install -r requirements.txt
```
# Prepare Dataset
To simpify the dataloading for your own dataset, we provide a default dataset that simply requires the path to the folder with your NifTI images inside, i.e.
```
root_dir/				# Path to the folder that contains the images
├── ID_001                  # ID is not important and can be randomly generated to ensure anonymity
        ├── T1C.nii.gz        # The sequence of the NifTI (strictly consistent)
        ├── T2F.nii.gz 
        ├── ROI.nii.gz  
├── ID_002     
        ├── T1C.nii.gz  
        ├── T2F.nii.gz 
        ├── ROI.nii.gz                 
├── ID_003  
        ├── T1C.nii.gz        
        |── ...           
├── ...                    
```
If needed, you may consider downloading the glioma public dataset from [BraTS 2023 Challenge](https://www.synapse.org/Synapse:syn51156910/wiki/).

Before training, the data needs to be preprocessed by **'Grayscale Normalization'** and mapped to the range of 0 to 255 by executing the following command
```
bash ./preprocess/Preprocess_grayscale_norm.sh
```


# Training
First, you need to train the condictional diffusion model. To do so in prepared dataset, you can run the following command:
```
python ./main/train_CBSI_gen.py
```
Second, you need to train the identiifcation model by runing the following command:
```
python ./main/train_CBSI_ide.py
```
Note that you need to provide the path to the dataset (e.g. `dataset.root_dir='/data/BraTS/BraTS 2023'`) to successfully run the command.

# Inference
In the inference stage, synthesis and identification are performed together. You can do this by running the following command:
```
python ./main/Inference.py
```
# Citation
To cite our work, please use
```
(To be updated)
