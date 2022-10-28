
This repository contains the code to reproduce the results in the RecSys 2022 paper
titled "Adversary or Friend? An Adversarial Approach to Improving Recommender Systems"
https://dl.acm.org/doi/abs/10.1145/3523227.3546784


wease.py --> runs a weighted version of EASE where the weights are derived from the user 
as described in the paper for EASE-IPW. When the parameter B is set to zero, wease.py is 
equivalent to EASE with uniform weights on users.

arm.py --> direct adversarial recommender model that usually does not work well.

r-larm.py --> rank loss based adversarial model with a linear model based adversary.

r-larm-nn.py --> rank loss based adversarial model with a two layer neural network based adversary.

We are providing a toy dataset to get started with this code. Since the datasets
used in the paper are third party data, we do not directly include those here.
The purpose of the dataset is to help you get started and to get an idea of the format in
which this code expects its inputs. The various hyper-parameters given in the files above
are in no way optimized for this toy dataset.

Following are the best hyper-parameters from the experiments in the paper that were found
through cross-validation. Models with these hyper-parameters were used for generating the 
plots in the paper.

ML20M:
EASE      : C=0.005, B=0.0
EASE-IPW  : C=0.002, B=0.6
R-LARM    : C=0.005, B=2000.0, lr=5e-5   
R-LARM(NN): C=0.005, B=500.0, lr=5e-5, dim=5

NFLX prize:
EASE      : C=0.002, B=0.0
EASE-IPW  : C=0.0005, B=0.8
R-LARM    : C=0.002, B=8000.0, lr=2e-5
R-LARM(NN): C=0.002, B=1000.0, lr=1e-5, dim=6

MSD:
EASE       : C=0.0002, B=0.0
EASE-IPW   : C=0.0002, B=0.4
R-LARM    :  C=0.0002, B=500.0, lr=1e-4
R-LARM(NN):  C=0.005,  B=500.0, lr=1e-4, dim=6
MSD batch_size was set at 8*1024 while the other two had a fixed batch size of 1024.
