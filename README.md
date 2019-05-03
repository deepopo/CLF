# CLF
Because of the need of scientific research, I had the honor to read *CLF* and tried to reproduce the paper. This Repository is a trial of reproduction of the paper:  
  >CLF: Zhang J , Yu P S . Integrated anchor and social link predictions across social networks[C] International Conference on Artificial Intelligence. AAAI Press, 2015.

The *CLF* can either predict social links among users in another network as well as anchor links aligning two networks.
Before to execute *CLF*, you should install the following packages:  
``pip install networkx``  
``pip install sklearn``  
The version of python is ``python==3.7.2`` and ``networkx==2.2``, ``sklearn==0.20.3``, but they are not mandatory unless it doesn't work.  
## Basic usage  
### Data  
We provide a *DBLP* dataset and its distributed copy in ./graph/ called ``G1`` and ``G2`` which are extracted from [Prado et al., 2013] to show the effect of *CLF*. The data are named as ``DBLP1.edges`` and ``DBLP2.edges`` respectively, in which each line consists of node ``ui`` and node ``uj`` within one network:  
``ui,uj``  
In addition, the ground truth alignments are also needed to compute the alignment accuracy. The file is named as ``DBLP.alignment`` in ./alignment/, in which each line consists of node ``ui`` in ``G1`` and node ``vi`` in ``G2``:  
``ui,vi``  

### Example  
In order to run *CLF* on the *DBLP* & distributed copy, execute the following command in ./src/:  
``python main.py --filename DBLP``  
If you need to modify the parameters, the complete execution command is (The parameters are optimal ones given in the paper):  
``python main.py --filename DBLP --align_train_prop 0.2 --alpha1 0.6 --alpha2 0.6 --c 0.1``  
You can check out the other options available to use with *CLF* using:  
``python main.py --help``  

### Evaluate
In order to evaluate the effect anchor links prediction, we use Top@30 and AUC to show the results.

If there are some factual errors, please let me know.
## Reference  
[1] Adriana Prado, Marc Plantevit, Celine Robardet, and J. F. Boulicaut. Mining graph topological patterns: Finding covariations among vertex descriptors. IEEE Transactions on Knowledge & Data Engineering, 25(9):2090–2104, 2013.  
[2] Zhang J , Yu P S . Integrated anchor and social link predictions across social networks[C] International Conference on Artificial Intelligence. AAAI Press, 2015.
