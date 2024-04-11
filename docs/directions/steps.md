## Run cxls_hitfinder:

### Description
For use at CXLS beamline, this software aims to identify the Bragg peaks, camera length (m), and photon energy (keV), of an image based on the training data. This software will be continuing development, so please check the most recent update to `main` branch for the most recent updates.

### 0. Installation

```bash
git clone https://github.com/adamkurth/cxls_hitfinder.git
```
or using GitLab:
    
```bash
```

Please install the following dependencies:

```bash
pip install -r requirements.txt
```

### 1. Directory Structure

The `checks.sh` script will ensure the proper directories are made in the `images/` directory. The `images/` directory is where the images will be stored. The `images/` directory will have the following structure:



```bash
cnn_hitfinder/
    ├── cnn/
    │   ├── src/
    │   │   ├── pkg/
    │   │   │   ├── __init__.py
    │   │   │   ├── util.py
    │   │   │   ├── models.py
    │   │   │   ├── functions.py
    │   │   ├── cnn.py
    │   │   ├── test.ipynb
    │   │   ├── etc.
    ├── images/
    │   ├── peaks/
    │   ├── labels/
    │   ├── peaks_water_overlay/
    │   └── water/
    ├── scripts/
    │   └── 
    ├── requirements.txt
    └── README.md
```

*Note that every folder in `images` contains `01` through `09` corresponding to the camlen/keV combination*

Parameter matrix for `camlen` and `keV`:

| Dataset (01-09) | camlen (m) | photon energy (keV) |
|---------------|------------|---------------------|
| `01`          | 1.5        | 6                   |
| `02`          | 1.5        | 7                   |
| `03`          | 1.5        | 8                   |
| `04`          | 2.5        | 6                   |
| `05`          | 2.5        | 7                   |
| `06`          | 2.5        | 8                   |
| `07`          | 3.5        | 6                   |
| `08`          | 3.5        | 7                   |
| `09`          | 3.5        | 8                   |

Contents of `images/`:
- `images/peaks` contains the Bragg peaks. 
- `images/labels` contains the labels for the Bragg peaks, (i.e. above a threshold of 5). 
- `images/peaks_water_overlay` contains the Bragg peaks overlayed with the respective keV water image. 
- `images/water` contains the different keV water images. 

The following bash script will make the needed directories in `images/`:

```bash
bash scripts/checks.sh
```

