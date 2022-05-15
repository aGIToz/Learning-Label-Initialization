# Learning-Label-Initialization
Source code for the paper "Learning Label Initialization for Time-Dependent Harmonic Extension". 
Arixv:https://arxiv.org/pdf/2205.01358.pdf
![2](https://user-images.githubusercontent.com/38216671/167074954-ce00bba4-1838-45c2-b5ff-f590e2dfa99b.png)

## Instructions
### harmonic_ext_learned_front
- This folder contains the scripts to reproduce the results for the PDE with learned front (\psi_0) and weights (and no weights).
- Navigate to the folder: 
```
cd harmonic_ext_learned_front
```
- For learning front and weigths, run: 
```
python run_cora_.py
```
- For learning only the front, run:
```
python run_cora_.py --nowgts
```
- The above commands should produce a `cora.txt` file with test scores over 100 iterations.

### harmonic_ext_include_val
- This folder contains the scripts to reproduce the results for the PDE with learned front and including the validation labels after the training.
- Navigate to the folder: 
```
cd harmonic_ext_include_val
```
- Run it: `
```
python run_cora_.py
```
- This should produce a `cora.txt` file with test scores over 100 iterations.
- You should see no variations for the citation graphs as the split is fixed.

## Dependencies
Any version of the following libaries should be fine, but the code has been tested for:
- pytorch==1.8.1
- torch-geometric==1.7.0
- torchdiffeq==0.2.2

## Tasks
- [x] Add the scripts for the harmonic extension with learned front and weights.
- [x] Write proper instruction to run the above item.
- [x] Add the scripts for the PDE with learned front and including the validation labels after the training.
- [x] Write proper instrcution to run the above item.
- [x] Create a flag to run with no weights.
- [x] Put an extended version of the paper on arxiv.
- [ ] Do an extra experiment for ogb-arxiv
