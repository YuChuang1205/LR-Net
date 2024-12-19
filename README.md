## The official complete code for paper "LR-Net: A Lightweight and Robust Network for Infrared Small Target Detection [[Paper/arXiv](https://arxiv.org/abs/2408.02780)]"

**The proposed scheme won the *2nd prize* in the "ICPR 2024 Resource-Limited Infrared Small Target Detection Challenge Track 1: Weakly Supervised Infrared Small Target Detection"**  

<p align="center">
  <img src="imgs/Overview of LR-Net.png" alt="Overview of LR-Net" width="800"/><br>  
</p> 

<p align="center">
  Our LR-Net abandons the complex structure and achieves an effective balance between detection accuracy and resource consumption. the parameters and FLOPs of LR-Net are only <b>0.020 M</b> and <b>0.063G</b>.
</p> 


## Datasets
The dataset used in this manuscript is the dataset provided by the competition. The dataset includes SIRST-V2, IRSTD-1K, IRDST, NUDT-SIRST NUDT-SIRST-Sea, NUDT-MIRSDT, and Anti-UAV. According to the requirements of the competition organizers, the competition data cannot be made public. some other public datasets can be considered for use： 
* **NUDT-SIRST** [[Original dataset](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt)] [[paper](https://ieeexplore.ieee.org/document/9864119)]
* **SIRST** [[Original dataset](https://github.com/YimianDai/sirst)] [[paper](https://ieeexplore.ieee.org/document/9423171)]
* **IRSTD-1k** [[Original dataset](https://drive.google.com/file/d/1JoGDGF96v4CncKZprDnoIor0k1opaLZa/view)] [[paper](https://ieeexplore.ieee.org/document/9880295)]
* **SIRST3** [[Original dataset](https://github.com/XinyiYing/LESPS)] [[paper](https://arxiv.org/pdf/2304.01484)]

## How to use our code
1. Preparing the dataset

   **According to the dataset path set in "train_model.py", please place the downloaded dataset in the corresponding folder.**

2. Creat a Anaconda Virtual Environment

    ```
    conda create -n LR-Net python=3.8 
    conda activate LR-Net
    ```

3. Configure the running environment
   
   ```
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   pip install PyWavelets -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install opencv-python==4.5.4.60 -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install opencv-python-headless==4.5.4.60  -i https://pypi.tuna.tsinghua.edu.cn/simple 
   pip install segmentation-models-pytorch  -i https://pypi.tuna.tsinghua.edu.cn/simple 
   pip install albumentations==1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install thop -i https://pypi.tuna.tsinghua.edu.cn/simple 
   pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple 
    ```

4. Training the model  
   
    ```
    python train_model.py
    ```

5. Testing the Model

   Since there are extremely large resolution images in the competition dataset, we implemented a sliding window cropping strategy in "test_model.py". For general datasets, you can set "patch_size" to a larg value to perform whole-image inference directly. In addition, we use the **adjustable sensitivity (AS) strategy** in "test_model.py". For the details of the AS strategy, please see [[Paper/arXiv](https://arxiv.org/abs/2407.20090)].

    ```
    python test_model.py
    ```

7. Performance Evaluation
   
   A "true_mask" folder can be created under the main project and put the true labels in it. Then, you can run the following command directly. Or, you can modify the corresponding image paths in the "cal_mIoU_and_nIoU.py" and "cal_Pd_and_Fa.py" accordingly.

    ```
    python cal_mIoU_and_nIoU.py
    python cal_Pd_and_Fa.py
    ```
    
## Results
* **Quantative Results on Competition Dataset**:
<p align="center">
  <img src="imgs/Results of Networks.png" alt="Results of Networks" width="800"/><br>  
</p> 


* **Ablation Experiment Results on Competition Dataset**:
<p align="center">
  <img src="imgs/Ablation results.png" alt="Ablation results" width="800"/><br>  
</p> 


* **Qualitative results of Ablation Experiment on Competition Dataset**:
<p align="center">
  <img src="imgs/Some visualizations of ablation experiments.png" alt="Some visualizations of ablation experiments" width="800"/><br>  
</p> 


* **More results on Competition Dataset**:
  
  For more results on Competition Dataset, please refer to the paper [[Paper/arXiv](https://arxiv.org/abs/2408.02780)]. 


## Citation

Please cite our paper in your publications if our work helps your research. <br>
BibTeX reference is as follows:
```
@misc{yu2024lrnetlightweightrobustnetwork,
      title={LR-Net: A Lightweight and Robust Network for Infrared Small Target Detection}, 
      author={Chuang Yu and Yunpeng Liu and Jinmiao Zhao and Zelin Shi},
      year={2024},
      eprint={2408.02780},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02780}, 
}
```

word reference is as follows:
```
Chuang Yu, Yunpeng Liu, Jinmiao Zhao, and Zelin Shi. LR-Net: A Lightweight and Robust Network for Infrared Small Target Detection. arXiv preprint arXiv:2408.02780, 2024.
```

## Other link
1. My homepage: [[YuChuang](https://github.com/YuChuang1205)]
2. "MSDA-Net" demo: [[Link](https://github.com/YuChuang1205/MSDA-Net)]
3. "Refined-IRSTD-Scheme-with-Single-Point-Supervision" demo: [[Link](https://github.com/YuChuang1205/Refined-IRSTD-Scheme-with-Single-Point-Supervision)]
4. "PAL Framework" demo：[[Link](https://github.com/YuChuang1205/PAL)]

 --- 
### _!!! Update (2024-12-18): Our latest research work: the Progressive Active Learning (PAL) framework for infrared small target detection with single point supervision has been open sourced [[paper](https://arxiv.org/abs/2412.11154)][[code](https://github.com/YuChuang1205/PAL)]. [Everyone is welcome to use it](https://github.com/YuChuang1205/PAL)._
 --- 
 
