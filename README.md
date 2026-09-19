# Python Package for the Time Series Analysis Course at LTH

## Installation

This package has been tested with **Python 3.12** and **3.14**.
Before running the project, make sure the following dependencies are installed in your Python environment (either global or within a virtual environment):

* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `statsmodels`
* `filterpy`
* `soundfile`
* `jupyter` (to run the notebooks)

You can install all of them with:

```bash
pip install -r TimeSeriesAnalysis-main/TimeSeriesAnalysis-main/requirements.txt jupyter
```

### Option 1: Clone the Repository (Recommended)

1. Install [Git](https://git-scm.com/).
2. Open Git Bash (or your preferred terminal) and navigate to the folder where you want to place the project:

   ```bash
   cd path/to/your/folder
   ```
3. Clone the repository:

   ```bash
   git clone https://github.com/andreasjak/TimeSeriesAnalysis.git
   ```

### Option 2: Download as ZIP

Alternatively, you can download the project folder as a ZIP file directly from GitHub and extract it to your chosen location.

Once installed or downloaded, you can run your projects directly from within the folder.


### Note

You can also install the package as a **local editable library** using `pip`:

```bash
pip install -e path/to/TimeSeriesAnalysis/TimeSeriesAnalysis-main/TimeSeriesAnalysis-main
```


## Contents

MATLAB is not needed; all course material is in Python.

* `Lectures/` – the example code from the lectures (`code1.ipynb`–`code28.ipynb`) and the examination code format (`examCode.py`).
* `labs/` – the three computer exercises (`lab1.ipynb`–`lab3.ipynb`).
* `mini_projects/` – the three voluntary mini-projects.
* `data/` – all data sets. See [`data/README.md`](./data/README.md) for how to load them.
* `TimeSeriesAnalysis-main/TimeSeriesAnalysis-main/tsa_lth/` – the course Python package, which replaces the MATLAB functions used in the course.

Start Jupyter from the repository folder (`jupyter notebook` or `jupyter lab`) and open the notebooks from there, so that the relative paths to `data/` and `tsa_lth` work.


## Contributing

Before contributing or pushing any code, please read the [`CONTRIBUTING.md`](./CONTRIBUTING.md) file.


