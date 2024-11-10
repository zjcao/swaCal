# swaCal

## Deep Neural Network Confidence Calibration from Stochastic Weight Averaging

<div align=left>
<img src=https://github.com/zjcao/swaCal/blob/main/_figures/swa.png width=50%/ >
</div>


### Requirements

The following function libraries should be installed, including ``python、numpy、matplotlib、tqdm`` and ``torch、torchvision``.


### Quick Start

- #### Custom Dataset

> Please modify the path settings(``data_dir``)  in ``./swaCal/data/cinic10.py/`` and ``cifar10.py`` files.

 
- #### Testing

```shell
python testing_cinic10.py
```


- #### Reliability diagram

<div align=left>
<img src=https://github.com/zjcao/swaCal/blob/main/_figures/cinic_10_ralia_before.png width=40%/> <img src=https://github.com/zjcao/swaCal/blob/main/_figures/cinic_10_ralia_after.png width=40%/>
</div>


### Citation
```BibTeX
@article{cao2024deep,
  title={Deep Neural Network Confidence Calibration from Stochastic Weight Averaging},
  author={Cao, Zongjing and Li, Yan and Kim, Dong-Ho and Shin, Byeong-Seok},
  journal={Electronics},
  volume={13},
  number={3},
  pages={503},
  year={2024},
  ISSN = {2079-9292},
  DOI = {10.3390/electronics13030503}
}
```
