# Structure Tensor Utility Functions

Tools for extraction and analysis of results from structure tensor algorithm. Compatible with the [2D/3D structure tensor library](https://github.com/Skielex/structure-tensor/tree/master) or with a proposed expansion for [scale-space structure tensor](https://github.com/PaPieta/stss).

Based heavily on Matlab implementation of similar functionalities by Hans Martin Kjer.


## Installation
1. Clone the repository.
2. Install the library

```
   cd stutil
   pip install .
```

# Prerequisites
Library requires a number of libaries, especially for enabling its full capabilities. The recommended list can be found in ```requrements.txt``` file and installed with the below command:
```sh
  pip install -r requirements.txt
```
With it, all necessary libraries are installed, including those only needed for running the demo fully and performing structure tensor analysis.

With ```requrements_minimal.txt```, only the libraries necessary for the execution of the provided functions are installed.

# Usage examples

Check provided ```demo.ipynb``` notebook for a step-by-step example of using the library on real data. To run the full example yourself, download the associated scan volume here: https://drive.google.com/file/d/1AqDlMG172BxqkMUJ6v9iyc0Y22ZixTzt/view?usp=sharing