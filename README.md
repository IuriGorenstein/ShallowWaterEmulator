# ShallowWaterEmulator
A Physics-Informing Shallow-Water Emulator using Lie AutoEncoders to construct a linear solver of the AB3-AM4 shallow water algorithm.

The code present in the SWE.py file was adapted from a previous code elaborated by Pedro S. Peixoto, professor in the Mathematical Institute of the University of São Paulo, São Paulo, Brazil. 
The original SWE.py code can be found at [https://www.ime.usp.br/pedrosp/](https://www.ime.usp.br/~pedrosp/modelagem-numerica-atmosfera/), in the Jupyter notebook from "Shallow water equations on the straight line and their discretization by differences/finite volumes on a shifted mesh" class.
The adaptations are:
+ Inclusion of the AB3-AM4 shallow water calculations in the tend functions from SWE_2D_num_method.
+ Neural network training functions.
+ Neural Network integrations of the shallow water variables.
+ Adaptation of plotting and diagnostics tools.
  
The remaining files and codes are part of the Ph.D thesis of Iuri Gorenstein in the Oceanographic Institute of the University of São Paulo, São Paulo, Brazil.

The training files of a LieAE have been uploaded to the repository (folder S26WE_pinnS4_dt+_ReLU6.0.1.0). It contains the figures from training and testing of the non-linear geostrophic equilibrium test case (network's best score), and the complete training logs (train_his.html), containing learning rate, and evolution of loss functions in the training and validation dataset.

The same pre-trained LieAE network analysed in the study has also been uploaded to the repository. Its name (S26WE_pinnS4_dt+_ReLU6.0.1.0) considers the SWE.py training function, the pinn encoder type (pinnS4), fixed resolution (dt+), with a three-layer encoder-decoder (X.X.0) of dimension 16*4 (0.X.X) and ReLU6 activation function. In the Lattent space, the linear operator holds dimension 20 (X.1.X).

Iuri Gorenstein.
