# ShallowWaterEmulator
A Physics Informing Shallow-Water Emulator using a Lie Auto-Encoder to construct a linear solver of the AB3-AM4 shallow water algorithm.

The code present in SWE.py file was addapted from a previous code elaborated by Pedro S. Peixoto, professor in the Mathematical Institute of University of São Paulo, São Paulo, Brazil. 
The original SWE.py code can be found at [https://www.ime.usp.br/pedrosp/](https://www.ime.usp.br/~pedrosp/modelagem-numerica-atmosfera/), in the jupyter notebook from "Shallow water equations on the straight line and their discretization by differences/finite volumes on a shifted mesh" class.
The adaptations are:
+ Inclusion of the AB3-AM4 shallow water tendecies calculations
+ Neural network training functions.
+ Neural Network integrations of the shallow water variables.

The remaining files are from Iuri Gorenstein. This code is part of the Ph.D thesis of Iuri Gorenstein in the Oceanographic Institute of University of São Paulo, São Paulo, Brazil.
