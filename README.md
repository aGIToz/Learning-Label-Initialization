# Learning-Label-Initialization
Source code for the paper "Learning Label Initialization for Time-Dependent Harmonic Extension". 
Arixv:https://arxiv.org/pdf/2205.01358.pdf
![2](https://user-images.githubusercontent.com/38216671/167074954-ce00bba4-1838-45c2-b5ff-f590e2dfa99b.png)

## Instructions
### harmonic_ext_learned_front
- This folder contains the scripts to reproduce the results for the PDE with learned front and weights (and no weights).
- To run it: `cd harmonic_ext_learned_front; python run_cora_.py`, this should produce a `cora.txt` file with test scores over 100 iteraitons

### harmonic_ext_include_val
- To produce the results of the PDE with learned front and weights and inlcuding the new labels (val labels)
- In progress.

## Dependencies
Any version of the following libaries should be fine, but the code has been tested for:
- pytorch==1.8.1
- torch-geometric==1.7.0
- torchdiffeq==0.2.2

## Tasks
- [x] Add code for harmonic extension with learned front, with and without weights.
- [x] Write proper instruction to reproduce the above item.
- [ ] Add code and instructions for including the val data after training [in progress].
