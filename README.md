#### E-QSMnet: Efficient Quantitative Susceptibility Mapping Using Knowledge Distillation in a DeepConvolutional Neural Network
###### Naveen Paluru and Phaneendra Yalavarthy
(in review, NeuroImage, April 2021)

## data preparation

Please refer to https://github.com/SNU-LIST/QSMnet for data downloading and pre-processing. Please cite the following if you are using the data or QSMnet model:

J. Yoon, E. Gong, I. Chatnuntawech, B. Bilgic, J. Lee, W. Jung, J. Ko, H. Jung, K. Setsompop, G. Zaharchuk, E.Y. Kim, J. Pauly, J. Lee. Quantitative susceptibility mapping using deep neural network: QSMnet. Neuroimage. 2018 Oct;179:199-206. [Link](https://www.sciencedirect.com/science/article/pii/S1053811918305378)

## python files

**trn.py** is the run file to train the model, **tst.py** is the inference file, **student.py** has EQSMnet, **teacher.py** has QSMnet and **myDataset.py** is data iterator. 
