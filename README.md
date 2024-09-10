# AutoEncodingBayesianInverseGames.jl

[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

This repository contains code for our paper at WAFR 2024: [Auto-Encoding Bayesian Inverse Games](https://arxiv.org/pdf/2402.08902). We propose an **end-to-end differentiable** pipeline that embeds a **differentiable Nash game solver** into a **generative model** for multi-modal uncertainty inference in multi-agent scenarios. For more information, please visit our [project website](https://xinjie-liu.github.io/projects/bayesian-inverse-games/).

To differentiate through a game solver, we analytically derive gradient from the optimality conditions of the game based on the implicit function theorem. For more details, please refer to our [prior work](https://xinjie-liu.github.io/projects/game/). 

<a href ="https://arxiv.org/abs/2402.08902"><img src="https://xinjie-liu.github.io/assets/img/liu2024wafr_teaser.png"></a>


<a href ="https://xinjie-liu.github.io/assets/pdf/liu2024auto.pdf"><img src="https://xinjie-liu.github.io/assets/img/liu2024auto.png" width = "560" height = "396"></a>

![wafr demo](https://xinjie-liu.github.io/assets/img/liu2024wafr_demo.gif =300x)

## How to use

### Initial setup

* Install Julia 1.10.5

* Set up a license key for the PATHSolver (see `License` section below)

* Start Julia in the root directory by typing `julia` in the terminal

* Activate the package environment by hitting `]` to enter the package mode first and then type: `activate .`

* Instantiate the environment in the package mode if you haven't done so before by typing `instantiate`

* Exit the package mode by hitting the backspace key; precompile the package: `using DrivingExample`

### Run simulation

Run traffic intersection simulation with a pre-trained VAE model in the `data\` folder:

```
DrivingExample.run_intersection_inference()
```

The generated videos will be stored in the `data\` folder 

### Run model training

Generate a traffic dataset:

```
DrivingExample.run_intersection_data_collection()
```

Run VAE training:

```
DrivingExample.train_generative_model_with_driving_data()
```

The generated objects will be stored in the `data\` folder

## Cite this work

```
@article{liu2024auto,
  title={Auto-Encoding Bayesian Inverse Games},
  author={Liu, Xinjie and Peters, Lasse and Alonso-Mora, Javier and Topcu, Ufuk and Fridovich-Keil, David},
  journal={arXiv preprint arXiv:2402.08902},
  year={2024}
}
```

## Acknowledgements

This project adapts the following packages as libraries:

* [Contingency Games](https://github.com/lassepe/peters2024ral-code)

* [DifferentiableAdaptiveGames.jl](https://github.com/xinjie-liu/DifferentiableAdaptiveGames.jl)

## License

This package uses PATH solver (via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl)) under the hood. Larger-sized problems require to have a license key. By courtesy of Steven Dirkse, Michael Ferris, and Tudd Munson, temporary license keys are available free of charge. For more details about the license key, please consult [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl) (License section). 

Note that when no license is loaded, PATH does **not** report an informative error and instead may just report a wrong result. Thus, please make sure that the license is loaded correctly before using the solver.

---

> TODO:
> Multi-processing part for accelerating large-scale training
