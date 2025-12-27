# Google Colab Setup Instructions

## Python Version
This project is tested with Python 3.12.12 on Google Colab.

## Installation Steps

### 1. Install System Dependencies (Required for Manim)

Run this in a Colab cell first:

```python
!sudo apt update
!sudo apt install -y libcairo2-dev ffmpeg \
    texlive texlive-latex-extra texlive-fonts-extra \
    texlive-latex-recommended texlive-science \
    tipa libpango1.0-dev
```

### 2. Install Python Dependencies

```python
!pip install -r requirements.txt
```

Or install individually:

```python
!pip install torch torchvision numpy h5py manim tqdm
```

### 3. Restart Runtime

After installation, restart the Colab runtime:
- Go to Runtime → Restart runtime

### 4. Verify Installation

```python
import torch
import manim
print(f"PyTorch: {torch.__version__}")
print(f"Manim: {manim.__version__}")
```

## Alternative: Using mscene for Easier Colab Integration

If you prefer a simpler setup, you can use `mscene` which handles Manim in Colab:

```python
%pip install -q mscene
import mscene
%mscene -l manim
from mscene.manim import *
```

However, the standard Manim installation should work fine with the system packages above.

## Notes

- Colab has limited disk space, so checkpoints may need to be saved to Google Drive
- Manim rendering can be slow on Colab's free tier - consider using lower quality for previews
- GPU is recommended for training but not required for visualization rendering

