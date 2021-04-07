#### E-QSMnet: Efficient Quantitative Susceptibility Mapping Using Knowledge Distillation in a DeepConvolutional Neural Network
###### Naveen Paluru and Phaneendra Yalavarthy
(in review, NeuroImage, April 2021)

## data preparation

Please refer to https://github.com/SNU-LIST/QSMnet for data downloading and pre-processing. Please cite the following if you are using the data or QSMnet model:

J. Yoon, E. Gong, I. Chatnuntawech, B. Bilgic, J. Lee, W. Jung, J. Ko, H. Jung, K. Setsompop, G. Zaharchuk, E.Y. Kim, J. Pauly, J. Lee. Quantitative susceptibility mapping using deep neural network: QSMnet. Neuroimage. 2018 Oct;179:199-206.

## python files

**trn.py** is the run file to train the model, **tst.py** is the inference file, **student.py** has EQSMnet, **teacher.py** has QSMnet and **myDataset.py** is data iterator. 



## GRaphical Abstract
<p align="center">
  <img src="https://github.com/NaveenPaluru/E-QSM/blob/main/graphics.png">
</p>



#### Any query, please raise an issue or contact :

*Dr. Phaneendra  K. Yalavarthy* 

*Assoc.Prof, CDS, IISc Bangalore, email : yalavarthy@iisc.ac.in*

*Naveen Paluru*

*(PhD) CDS, MIG, IISc Bangalore,  email : naveenp@iisc.ac.in*
