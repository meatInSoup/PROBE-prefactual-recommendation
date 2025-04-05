# PROBE
This repository contains the code to replicate the experiments of the paper "Prefactual Recommendation by Bayesian Preference Elicitation".

1. Train MLP classifiers

```sh
python train_model.py --clf mlp --data german bank student compas adult gmc --num-proc 16
```

2. Run experiments

- Experiments under correctly specified cost function (Sec 6.1)
```sh                                             
python run_expt.py -e 1 --datasets german bank student compas adult gmc -clf mlp --methods probe face reup_graph -uc
```
- Experiments under cost function misspecification (Sec 6.2)
```sh                                             
python run_expt.py -e 2 --datasets german bank student compas adult gmc -clf mlp --methods probe face reup_graph -uc
```
