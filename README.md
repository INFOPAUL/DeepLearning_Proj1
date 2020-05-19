# Mini-project 1 - Classification, weight sharing, auxiliary losses

Proj description  
https://fleuret.org/ee559/materials/ee559-miniprojects.pdf

You can find the guidance below for going through the folders in our repo: 
# Architectures: 
In architectures folder, we have following models for our neural networks. 
- Linear.py	
- Logistic.py
- NoWeightSharing.py
- NoWeightSharing_AuxLosses.py
- SimpleConvNet.py
- WeightSharing.py
- WeightSharing_AuxLosses.py

# dataset:
In dataset folder, you can see the python script for creating the dataset that we used in training and test. 
- CustomDataset.py


# Running model in VM
In order to run our model in VM, you type `python test.py`. This will give result for the one network defined above (by default it is NETWORK=1). You can change the 'NETWORK' parameter inside test.py according to the network model that you wanna run. You can find the values for that parameter as following: 
1 - SimpleConvNet
2 - WeightSharing
3 - WeightSharingAuxLosses
4 - NoWeightSharing
5 - NoWeightSharingAuxLosses

