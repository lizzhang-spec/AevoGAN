# AevoGAN
A Method based on Evolutionary Algorithms and Channel Attention Mechanism to enhance Cycle Generative Adversarial Network performance for Image Translation.
![framework.png](https://s2.loli.net/2023/02/27/OVSgl72GAmkIuJR.png)

# Descriptions
Requirements: Python 3.6, PyTorch 0.4
<br />
Dataset: [Horse2Zebra](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset)
- **datasets.py:** data sampling and pre-processing
- **GA.py:** definitions of selection, crossover and mutation operations in evolutionary algorithms
- **models.py:** design of generators and discriminator networks
- **search.py:** the process of searching for the best performing generator
- **train.py:** the adversarial training between generators and discriminators
- **test.py:** test file for compressed networks
- **utils.py:** necessary tools to assist the implementation

# Performance
The results of experiments on horse2zebra datasets. The generated images by methods CycleGAN, AevoGAN+SA (Spatial Attention), and AevoGAN are presented sequentially from left to right. 
![results.jpg](https://s2.loli.net/2023/02/27/bdkpyoFOuCVEGXl.jpg)

# Citation
Xue Y, Zhang Y, Neri F. A Method based on Evolutionary Algorithms and Channel Attention Mechanism to Enhance Cycle Generative Adversarial Network Performance for Image Translation[J]. International Journal of Neural Systems, 2023: 2350026-2350026.
