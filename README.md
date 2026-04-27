#  Galaxy Classifier — CNN with PyTorch

  

A Convolutional Neural Network (CNN) built from scratch in PyTorch to classify galaxy images into three morphological categories: **Elliptical**, **Spiral**, and **Barred Spiral**.

  

---

  

##  Project Overview

  

This project trains a custom CNN (inspired by the [TinyVGG](https://poloclub.github.io/cnn-explainer/) architecture) on a curated subset of the [Galaxy Zoo 2 dataset](https://www.kaggle.com/datasets/robertmifsud/resized-reduced-gz2-images) from Kaggle. The goal is to learn to distinguish between the three most common galaxy morphologies from optical imaging data.

  

---

  

##  Dataset

  

-  **Source:** [Resized/Reduced GZ2 Images — Kaggle](https://www.kaggle.com/datasets/robertmifsud/resized-reduced-gz2-images) (derived from Galaxy Zoo 2 / SDSS)

-  **Classes:** Barred Spiral, Elliptical, Spiral

-  **Split:**

- Training: 5,000 images per class (15,000 total)

- Testing: 1,000 images per class (3,000 total)

-  **Total:** ~18,000 images

- Data is downloaded at runtime from a Dropbox link and extracted automatically.

  

---

  

##  Model Architecture

  

The model `GalaxyClass` is a custom CNN with three convolutional blocks followed by a fully connected classifier.

  

```

Input (3, 224, 224)

│

▼

Conv Block 1 — Conv2d → ReLU → Conv2d → ReLU → MaxPool2d

│

▼

Conv Block 2 — Conv2d → ReLU → Conv2d → BatchNorm2d → Dropout2d(0.2) → ReLU → MaxPool2d

│

▼

Conv Block 3 — Conv2d → ReLU → Conv2d → ReLU → MaxPool2d

│

▼

Classifier — Flatten → Linear(25088, 512) → Dropout(0.5) → ReLU → Linear(512, 3)

│

▼

Output (3 logits)

```

  

-  **Hidden units:** 32 (per conv layer)

-  **Loss function:** CrossEntropyLoss

-  **Optimizer:** Adam (`lr=0.0004`)

-  **Epochs:** 40

  

---

  

##  Data Preprocessing & Augmentation

  

Images are resized and augmented before being fed to the model:

  

```python

transforms.Compose([

transforms.Resize((224, 224)),

transforms.RandomHorizontalFlip(),

transforms.RandomRotation(20),

transforms.ToTensor()

])

```

  

---

  

##  Requirements

  

Install dependencies with:

  

```bash

pip  install  torch  torchvision  matplotlib  pillow  numpy  scikit-learn  tqdm  torchinfo  requests

```

  

Or use Google Colab (recommended for GPU access).

  

---

  

## Usage

  

1.  **Clone the repository:**

```bash

git clone https://github.com/dev-tiwari-404/Galaxy-Image-Classifier.git

cd galaxy-classifier

```

  

2.  **Open the notebook:**

```bash

jupyter notebook Galaxy_Classifier_nn_5.ipynb

```

  

3.  **Run all cells.** The notebook will:

- Automatically download and extract the dataset

- Build and display the model summary

- Train for 20 epochs, printing loss/accuracy and a confusion matrix each epoch

- Run a sample prediction at the end

  

> ⚡ A CUDA-compatible GPU is strongly recommended. The notebook includes device-agnostic code that falls back to CPU automatically.

  

---

  

##  Evaluation

  

After each epoch, the notebook prints:

- Training loss & accuracy

- Test loss & accuracy

- A **confusion matrix** (Barred Spiral / Elliptical / Spiral) via `sklearn` and `matplotlib`

  

A final single-image prediction demo is included at the end of the notebook.

  

---

  

##  Project Structure

  

```

galaxy-classifier/

├── Galaxy_Classifier_nn_5.ipynb # Main notebook

├── README.md

└── data/ # Auto-created on first run

└── Galaxy_E_S_SB/

├── train/

│ ├── Barred_Spiral/

│ ├── Elliptical/

│ └── Spiral/

└── test/

├── Barred_Spiral/

├── Elliptical/

└── Spiral/

```

  

---

  

##  Background & Motivation

  

Galaxy morphological classification is a fundamental task in observational astronomy. Projects like [Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) have used crowdsourced human classification to label hundreds of thousands of galaxies — this project explores automating that with deep learning.

  

The dataset used here is a small, curated subset of the GZ2 dataset, originally sourced from the Sloan Digital Sky Survey (SDSS).

  

---

  

##  Notes & Known Limitations

  

- The model currently handles only 3 galaxy classes. Future work may expand to finer-grained GZ2 labels.

- The dataset download depends on an external Dropbox link — if it goes down, you'll need to source the images from Kaggle directly.

- The architecture is intentionally kept simple (TinyVGG-style) as a learning exercise. Transfer learning (e.g. ResNet, EfficientNet) would likely achieve higher accuracy.

  

---

  

##  License

  

This project is for educational and research purposes. Galaxy Zoo data is made available by the [Zooniverse](https://www.zooniverse.org/) community under their respective terms.

  

---

  

##  Acknowledgements

  

- [Galaxy Zoo / Zooniverse](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) for the original dataset

- [CNN Explainer (TinyVGG)](https://poloclub.github.io/cnn-explainer/) for the architecture inspiration

- [mrdbourke's PyTorch course](https://www.learnpytorch.io/) — a great resource for learning PyTorch
