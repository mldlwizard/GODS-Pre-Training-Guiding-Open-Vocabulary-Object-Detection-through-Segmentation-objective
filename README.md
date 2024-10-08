# YALALA MOHIT

# GODS pre-training : Guiding Object detection through segmentation

## Overview

This repository contains an enhanced training strategy for OVOD models validated on the [Scaling Open-Vocabulary Object Detection](https://arxiv.org/pdf/2306.09683v2.pdf).

## Table of contents

- [GODS pre-training](#GODS pre-training : Guiding Object detection through segmentation)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download datasets & Installation](#download-datasets)
    - [How Test and Train](#how-test-and-train)
      - [Test](#test)
      - [Evaluate model](#Evaluate-model)
      - [Resume train model](#resume-train-model)

## Download datasets & Installation

The [LVIS](https://www.lvisdataset.org/dataset) dataset site can be used to download the data.
However, to get things working, you only need to download the annotation files, and update the ANNOTATION_PATH in the trainer.py or evaluator.py, depending on if you want to train the model, or perform evaluation using an existing model. The IMAGE_DIR path need not have the LVIS images, as the dataloader handles their downloading to the given path.

The environment details are stored in requirements.txt and requirements_pip.txt
Use them in a virtual environment, to set up the required environment
After that, Install the LVIS API.
```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment

# clone the lvis repository
git clone https://github.com/lvis-dataset/lvis-api.git
cd lvis-api
# install COCO API. COCO API requires numpy to install. Ensure that you installed numpy.
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# install LVIS API
pip install .
# test if the installation was correct
python test.py
# Work for a while ...
deactivate  # Exit virtual environment
```

## How Test and Train

### Train model
The training script can be found in trainer.py
The hyperparameters can be updated at the end of the file.
Update the Hyper-parameters and paths as per wish and requirement.
To run training,
```bash
    !python3 trainer.py
```

## Evaluate Model
The evaluation script can be found in evaluator.py
The hyperparameters can be updated at the end of the file.
Update the Hyper-parameters and paths as per wish and requirement.
To run evaluation,
```bash
    !python3 evaluator.py
```

### Resume Train model
The training script can be found in trainer.py
The hyperparameters can be updated at the end of the file.
Update the Hyper-parameters and paths as per wish and requirement.
The RESUME_TRAINING flag is set to True. If there exists a last_saved model in the save_dir folder,
the script will automatically resume the training.
To run training,
    !python3 trainer.py
