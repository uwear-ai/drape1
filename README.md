# Drape1

Drape1 is a Python package for on model garment generation. This repository contains all the necessary components to run Drape1.

## Installation

1. First, ensure you have Python 3.8 or higher installed on your system.

2. Clone this repository:
```bash
git clone https://github.com/uwear-ai/drape1.git
cd drape1
```

3. Install the package and its dependencies:
```bash
pip install -e .
```

This will install the following dependencies:
- einops
- diffusers
- transformers
- torch
- torchvision
- accelerate
- transparent-background

## Usage

The package provides several modules for garment draping and visualization:

- `pipeline.py`: Main pipeline for garment draping
- `unet_base.py`: Base UNet model implementation
- `unet_ref.py`: Reference UNet model implementation
- `utils.py`: Utility functions for the project

To use Drape1 in your Python code:

```python
from src.pipeline import load_drape, infer_drape
from transparent_background import Remover
from PIL import Image

# Initialize the pipeline
pipeline = load_drape()
remover = Remover()

prompt = "A woman posing for a photoshoot, wearing a 'uwear.ai' white tshirt"
image_ref = Image.open("assets/uwear_tshirt.png").convert("RGB")

# Generate an image
image = infer_drape(
    pipe=pipeline,
    prompt=prompt, 
    image_ref=image_ref, 
    remover=remover,
    seed=42
)

image

```

## Requirements

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended for better performance)

If you have issues with bitsandbytes on Windows, try using an older version, such as 0.43.0

## License

This project is licensed under the terms of the APACHE 2.0 license.

## Author

Axel Havard from uwear.ai 