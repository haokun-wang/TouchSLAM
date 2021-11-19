# TouchSLAM
We varified our code with the following configuration:
```
OS: Windows10
Python version: 3.7.10
GPU: GeForce RTX 3090 (training)
```
## Requirements
The dependencies of TouchSLAM are showed as below:
- Python packages
  - numpy==1.20.3
  - opencv_python==4.5.3.56
  - metrics==0.3.3
  - torch=1.9.0+cu111
  - torchvision==0.10.0+cu111

Before using TouchSLAM, please ensure you have installed all above packages. You can install them simply by
```
python -m pip install -r requirements.txt
```

## Results
You can follow the instructions in `visualization.ipynb` step by step to reproduce our results.