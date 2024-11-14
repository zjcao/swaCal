# swaCal

## Deep Neural Network Confidence Calibration from Stochastic Weight Averaging

<div align=left>
<img src=https://github.com/zjcao/swaCal/blob/main/_figures/swa.png width=50%/ >
</div>


### Requirements

The following libraries should be installed, including 

``python、numpy、matplotlib、tqdm、scikit-learn``

and 

``torch、torchvision、torchmetrics``.


### Quick Start

- #### Custom dataset

> Please modify the path settings(``data_dir``)  in ``./swaCal/data/cinic10.py/`` and ``cifar10.py`` files.


- #### Training

```sh
python training_cinic10.py
```

- #### Testing

```python
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

### Acknowledgements

We thank the authors of *** for their [code]().

> Autor: czjing (last edited: Nov10, 2024) \
> ![](https://komarev.com/ghpvc/?username=zjcao&abbreviated=true&label=Visits)
