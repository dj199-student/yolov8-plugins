# Contributing to YOLOv8 Plugins

Thank you for your interest in contributing! This document outlines how to contribute to this project.

## Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd yolo-v8

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## Adding a New Plugin

Plugins are the core of this project. Follow these steps to add a new one:

### 1. Choose the Right Category

| Category | Directory | Examples |
|----------|-----------|----------|
| Attention | `models/plugins/attention/` | SE, CBAM, ECA, CA |
| Convolution | `models/plugins/conv/` | GhostConv, PConv, RepConv |
| Transformer | `models/plugins/transformer/` | ViT Block, MHSA |
| Neck | `models/plugins/neck/` | BiFPN, ASFF |
| SPP | `models/plugins/spp/` | SPPCSPC, ASPP |
| Head | `models/plugins/head/` | DyHead |

### 2. Implement Your Plugin

Create a new `.py` file in the appropriate directory:

```python
"""
MyPlugin: Brief description of your module.
Reference: [Paper Title](https://arxiv.org/abs/...)
"""

import torch
import torch.nn as nn
from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "my_plugin",              # Unique name (snake_case)
    category="attention",      # Category
    description="One-line description of what it does",
)
class MyPlugin(nn.Module):
    def __init__(self, in_channels: int, param1: int = 16, **kwargs):
        super().__init__()
        # Your implementation here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your forward pass here
        return x


# Test block (REQUIRED)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)
    model = MyPlugin(in_channels=64).to(device)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Shape match: {x.shape == y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. Register the Plugin

Add your plugin to the `__init__.py` in the corresponding directory:

```python
from .my_plugin import MyPlugin
```

### 4. Test Your Plugin

```bash
# Run the built-in test
python models/plugins/attention/my_plugin.py

# Run existing tests
python tests/test_phase2_features.py
```

### 5. Update Documentation

- Add your plugin to `README.md` plugin table
- Add usage example to `tutorials/plugin_guide.md`
- Add a sample config in `configs/plugins/`

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Use Chinese comments for explanations, English for code identifiers
- Include type hints for all function signatures
- Keep files focused — one plugin class per file (except paired sub-modules)
- Every plugin must include a `if __name__ == "__main__":` test block

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your changes following the guidelines above.
3. Add/update tests if applicable.
4. Update documentation.
5. Submit a PR using the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
6. Ensure all checks pass.

## Reporting Issues

Use the issue templates:
- [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md)
- [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md)

## Questions?

Feel free to open a Discussion or Issue if you have questions.
