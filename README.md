# DynamicFL
**Towards Fairness-aware and Privacy-preserving Enhanced Collaborative Learning for Healthcare** [ Under Review]

![image](overview.jpg)

## **Getting Started**

### **1. Data**
This project uses three publicly available datasets. Please follow the steps below to download and prepare the datasets:

#### **Dataset 1: ChestXray**  
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data).  
2. Create a folder named `data` in the root directory of this project.  
3. Extract the downloaded dataset into the `data/covid` directory.  
   - After extraction, the folder structure should look like:  
     ```
     data/
       COVID-19/
         COVID/
         Normal/
         Lung_Opacity/
         Viral Pneumonia/
     ```  
4. Inside each of the class folders (`COVID`, `NORMAL`, `LUNG_OPACITY`, `VIRAL_PNEUMONIA`), **delete the `masks` subfolder** because this project focuses only on classification tasks and does not involve segmentation.  

---

#### **Dataset 2 & Dataset 3**  
1. Download the second and third datasets from [Zenodo](https://zenodo.org/records/10519652).  
2. Extract these datasets into the `data` directory.


### **2. Install Dependencies**
To set up the environment, ensure you have Python and Anaconda installed, then install the required libraries using the `requirements.txt` file, 12.1 cuda environment is recommended:

```bash
conda create -n dyfl python=3.9
conda activate dyfl
pip install -r requirements.txt
pip install --upgrade wandb protobuf

```
---

## **3. Usage**

**3.1 Training with Different ViT Architectures**

You can train the model with different ViT (Vision Transformer) architectures by specifying the --model parameter. Below are some examples:

```python

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_tiny --device cuda:0

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_small --device cuda:0

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_1B --device cuda:0

```

**3.2 Training with Non-IID Data Partitions**

You can simulate non-IID (non-Independent and Identically Distributed) data by using the --partition parameter and setting the Dirichlet distribution parameter (--dir). Below are some examples:


```python

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --partition dir --dir 0.1

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --partition dir --dir 0.3

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --partition dir --dir 0.5

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --partition dir --dir 0.7

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --partition dir --dir 0.9
```

**3.3 Training with Different Device Ratios**

You can allocate computational resources across devices in specific ratios using the --device_ratio parameter. Below are some examples:

```python

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --device_ratio 7:2:1

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --device_ratio 5:2:3

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --device_ratio 4:1:5

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --device_ratio 4:3:3

python dyfl_vit.py --dataset bloodcell --node_num 30 --model Vit_Base --device cuda:0 --device_ratio 3:6:1

```

**3.4 Training with Different Device Counts**

You can vary the number of devices (or nodes) in your distributed setup using the --node_num parameter. Below are some examples:

```python

python dyfl_vit.py --dataset bloodcell --node_num 90 --model Vit_Base --device cuda:0

python dyfl_vit.py --dataset bloodcell --node_num 180 --model Vit_Base --device cuda:0

python dyfl_vit.py --dataset bloodcell --node_num 360 --model Vit_Base --device cuda:0

```

**3.5 Running on CPU Only (High Memory Requirement)**

If you want to run only on CPU, you can set the --device parameter to cpu. Note: Running on CPU requires very high RAM, especially for larger ViT models. It is recommended to use ViT-Tiny for testing on CPU, with at least 128GB of RAM.

```python

python dyfl_vit.py --dataset bloodcell --device cpu --model vit_tiny --wandb 0

```
---

## **4. Contact**

For any questions or issues, please open an issue on the GitHub repository.




