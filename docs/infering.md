---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Inference

The models are saved in a `pt` PyTorch file format, and have specifically been saved such that their weights and biases (learned parameters), as well as the model architecture are all stored in a single file.

This file is then logged to Comet. 

In the Comet interface, if you are happy with a model's results you can register a version of the model along with a description. This is save the model and the associated training information with it. Under the training run with the saved model, you can access "Registry" -> "model version" -> API

Assuming you've correctly set your API keys in your current shell, you can download the registered models using Python:

```python
from comet_ml.api import API

api = API() 
api.download_registry_model("nannau-uvic", "name_of_model_registry", version="1.0.0", output_path="./", expand=True, stage=None)
```

This will download the model file locally so that you can perform inference easily.

Below is a quick snippet for how to perform inference. Actual inference will depend on what question you're trying to answer, computational hardware available to you, and so on.

```python
import torch
G = torch.jit.load("generator.pt")
x = torch.randn(1, 6, 64, 64).cuda() # lr climatex fields
hr_cov = torch.randn(1, 1, 512, 512).cuda() # hr invariant covariates
y = G(x, hr_cov)
```

Random placeholders are used here instead of actual data. However, if used with `nc2pt`, this data can be easily generated.