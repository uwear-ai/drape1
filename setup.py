from setuptools import find_packages, setup

setup(
    name="drape1",
    packages=find_packages(),
    install_requires=[
        "einops",
        "diffusers~=0.33",
        "transformers~=4.51",
        "torch",
        "torchvision",
        "accelerate~=1.6",
        "transparent-background",
        "bitsandbytes",
    ],
    version="1.0.0",
    description="This repo contains all you need to run Drape1",
    author="Axel Havard from uwear.ai",
)