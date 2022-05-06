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


## Tasks
- [x] Add code for harmonic extension with learned front, with and without weights.
- [ ] Write proper instruction to run the code [in progress].
- [ ] Add code for including the val data after training [in progress].
