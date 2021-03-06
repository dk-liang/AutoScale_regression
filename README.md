# AutoScale_regression
* An officical implementation of AutoScale regression-based method, you can find localization-based method from [here](https://github.com/dkliang-hust/AutoScale_localization). 
* [AutoScale](https://arxiv.org/abs/1912.09632) leverages a simple yet effective Learning to Scale (L2S) module to cope with signiﬁcant scale variations in both localization and regression.<br />

# Structure
```
AutoScale_regression
|-- data            # generate target
|-- model           # model path 
|-- README.md       # README
|-- centerloss.py           
|-- config.py          
|-- dataset.py       
|-- find_contours.py           
|-- fpn.py         
|-- image.py
|-- make_npydata.py
|-- rate_model.py
|-- val.py          
```

# Visualizations
![avatar](./images/result.png)

# Environment
python >=3.6 <br />
pytorch >=1.0 <br />
opencv-python >=4.0 <br />
scipy >=1.4.0 <br />
h5py >=2.10 <br />
pillow >=7.0.0

# Datasets
* Download ShanghaiTech dataset from [Baidu-Disk](https://pan.baidu.com/s/15WJ-Mm_B_2lY90uBZbsLwA), passward:cjnx ; or [Google-Drive](https://drive.google.com/file/d/1CkYppr_IqR1s6wi53l2gKoGqm7LkJ-Lc/view?usp=sharing)
* Download UCF-QNRF Dataset from  [Google-Drive](https://www.crcv.ucf.edu/data/ucf-qnrf/)
* Download JHU-CROWD ++  dataset from [here](http://www.crowd-counting.com/)
* Download NWPU-CROWD dataset from [Baidu-Disk](https://pan.baidu.com/s/1Rm1WTBcz3h6k5ZvLnCCiMA), passward:9z97; or [Google-Drive](https://drive.google.com/file/d/14m2Y6A9_Vq8yUcKvozXGBP4Gwoz3Dotx/view?usp=sharing)
# Generate target
```cd data```<br />
Edit "distance_generate_xx.py" to change the path to your original dataset folder.<br />
```python density_generate_xx.py```

“xx” means the dataset name, including sh, jhu, qnrf, and  nwpu.

# Model
Download the pretrained model from [Baidu-Disk](https://pan.baidu.com/s/1kAhMdfVCO6A4SgrYi0_sog), passward:9qfc ; or [Google-Drive](https://drive.google.com/drive/folders/1GPUG-JDIlEyFqOnsvDa71fjxiOl55x_Z?usp=sharing)
# Quickly test
* ```git clone https://github.com/dk-liang/AutoScale_regression.git```<br />
  ```cd AutoScale```<br />

* Download Dataset and Model

* Generate target

* Generate images list

  Edit "make_npydata.py" to change the path to your original dataset folder.<br />
  Run ```python make_npydata.py  ```.

* Test <br />
```python val.py  --test_dataset qnrf  --pre ./model/QNRF/model_best.pth --gpu_id 0```<br />
```python val.py  --test_dataset jhu  --pre ./model/JHU/model_best.pth --gpu_id 0```<br />
```python val.py  --test_dataset nwpu  --pre ./model/NWPU/model_best.pth --gpu_id 0```<br />
```python val.py  --test_dataset ShanghaiA  --pre ./model/ShanghaiA/model_best.pth --gpu_id 0```<br />
```python val.py  --test_dataset ShanghaiB  --pre ./model/ShanghaiB/model_best.pth --gpu_id 0```<br />
More config information is  provided in ```config.py  ```

# Training
coming soon.

# References
If you are interested in AutoScale, please cite our work:
```
@article{xu2019autoscale,
  title={AutoScale: Learning to Scale for Crowd Counting},
  author={Xu, Chenfeng and Liang, Dingkang and Xu, Yongchao and Bai, Song and Zhan, Wei and Tomizuka, Masayoshi and Bai, Xiang},
  journal={arXiv preprint arXiv:1912.09632},
  year={2019}
}
```
and
```
@inproceedings{xu2019learn,
  title={Learn to Scale: Generating Multipolar Normalized Density Maps for Crowd Counting},
  author={Xu, Chenfeng and Qiu, Kai and Fu, Jianlong and Bai, Song and Xu, Yongchao and Bai, Xiang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8382--8390},
  year={2019}
}
```


