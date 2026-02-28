# Face Generation with Generative Adversarial Networks (GANs)

A Deep Learning Nanodegree (DLND) project implementing Generative Adversarial Networks to generate realistic face images using TensorFlow 1.x.

## Project Overview

This project demonstrates the implementation of a Generative Adversarial Network (GAN) to generate synthetic face images. The project trains on two datasets sequentially:
1. **MNIST** - For initial testing and validation
2. **CelebA** - For generating realistic celebrity faces

The implementation uses a competition between a **Generator** and **Discriminator** network to create increasingly realistic fake images.

## Dataset Details

### MNIST
- **Source**: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Description**: 70,000 images of handwritten digits (0-9)
- **Format**: Grayscale, 28×28 pixels, single channel (1 channel)
- **Purpose**: Initial testing of the GAN architecture

### CelebA
- **Source**: [CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Description**: 200,000+ celebrity face images
- **Format**: RGB color images, 28×28 pixels (cropped and resized), 3 channels
- **Processing**: Images are cropped to remove non-face regions, then resized to 28×28
- **Preprocessing**: All images are normalized to range [-0.5, 0.5]

## Architecture

### Discriminator Network
Classifies images as real or fake using convolutional layers:
- **Input**: 28×28×channels image
- **Layers**: 3 convolutional blocks with batch normalization
- **Activation**: Leaky ReLU (alpha=0.01)
- **Output**: Sigmoid probability and logits
- **Stride Pattern**: 2x downsampling at each conv layer
- **Reusable**: Uses `tf.variable_scope('discriminator')` for weight reuse

### Generator Network
Creates synthetic images from random noise:
- **Input**: z vector (random noise, default 100 dimensions)
- **Layers**: Dense layer → Reshape → 3 transposed convolutional blocks
- **Activation**: Leaky ReLU for hidden layers, Tanh for output
- **Output**: 28×28×out_channel_dim image
- **Reusable**: Uses `tf.variable_scope('generator')` for weight reuse
- **Training Mode**: Batch normalization switches between training and inference

## Key Components

### 1. model_inputs()
Creates TensorFlow placeholders for:
- **input_real**: Real image batch (None, image_width, image_height, image_channels)
- **input_z**: Random noise vector (None, z_dim)
- **learning_rate**: Training learning rate (scalar)

### 2. discriminator()
- Takes images and an optional `reuse` parameter
- Returns tuple: (sigmoid output, logits)
- Progressively downsamples image with strided convolutions
- Uses batch normalization for training stability

### 3. generator()
- Takes noise vector `z` and output channel dimension
- Includes `is_train` parameter for batch norm mode switching
- Returns generated image tensor
- Progressively upsamples from 4×4 to 28×28

### 4. model_loss()
Implements adversarial loss with label smoothing:
- **Discriminator Loss**: BCE loss for real images (smoothed) + BCE loss for fake images
- **Generator Loss**: BCE loss trying to fool the discriminator
- Uses sigmoid cross-entropy for numerical stability

### 5. model_opt()
Creates optimization operations:
- Separates trainable variables by scope (discriminator vs generator)
- Returns separate optimizers for each network
- Uses Adam optimizer with customizable beta1 parameter

## Project Structure

```
DLND-face-generation/
├── dlnd_face_generation.ipynb      # Main Jupyter notebook with implementation tasks
├── dlnd_face_generation.html       # HTML export of the notebook
├── helper.py                       # Utility functions for data handling
├── problem_unittests.py            # Unit tests for validating implementations
└── README.md                       # This file
```

## File Descriptions

### helper.py
Provides utility functions:
- `download_extract()`: Downloads and extracts MNIST or CelebA datasets
- `get_image()`: Loads and preprocesses individual images
- `get_batch()`: Creates batches of images
- `images_square_grid()`: Visualizes images in a grid layout
- `Dataset` class: Manages batching and normalization
- `DLProgress`: Progress bar for downloads

### problem_unittests.py
Contains unit test functions:
- `test_model_inputs()`: Validates placeholder shapes and types
- `test_discriminator()`: Checks discriminator architecture and variable scope
- `test_generator()`: Checks generator architecture and variable scope
- `test_model_loss()`: Validates loss tensor shapes
- `test_model_opt()`: Validates optimizer creation
- `TmpMock` class: Context manager for mocking TensorFlow functions

### dlnd_face_generation.ipynb
Jupyter notebook containing:
- Dataset exploration and visualization
- Network implementation tasks
- Training loop setup
- Generated image visualization

## Requirements

- **TensorFlow**: >= 1.0 (uses TensorFlow 1.x API)
- **NumPy**: Array operations
- **PIL/Pillow**: Image processing
- **Matplotlib**: Visualization
- **tqdm**: Progress bars
- **Python 3.6+**

## How to Use

### 1. Setup
```bash
# Install dependencies
pip install tensorflow numpy pillow matplotlib tqdm

# Create data directory
mkdir -p ./data
```

### 2. Download Data
The notebook automatically handles dataset downloading via the helper module:
```python
import helper
helper.download_extract('mnist', './data')
helper.download_extract('celeba', './data')
```

### 3. Run the Notebook
```bash
jupyter notebook dlnd_face_generation.ipynb
```

### 4. Implementation Steps
Following the notebook, implement:
1. **model_inputs()** - Create input placeholders
2. **discriminator()** - Build discriminator network
3. **generator()** - Build generator network
4. **model_loss()** - Calculate adversarial losses
5. **model_opt()** - Create optimizers
6. **train()** - Implement training loop

### 5. Validation
Unit tests are provided to validate each component:
```python
tests.test_model_inputs(model_inputs)
tests.test_discriminator(discriminator, tf)
tests.test_generator(generator, tf)
tests.test_model_loss(model_loss)
tests.test_model_opt(model_opt, tf)
```

## Training Details

### Preprocessing
- Images normalized to range [-0.5, 0.5]
- MNIST: Single grayscale channel
- CelebA: Three RGB channels

### Hyperparameters (Typical)
- **Batch Size**: 32-64
- **Learning Rate**: 0.0002 (Adam optimizer)
- **Beta1**: 0.9 (Adam momentum parameter)
- **Z Dimension**: 100 (noise vector size)
- **Label Smoothing**: 0.1 (for discriminator)

### Loss Functions
- **Discriminator**: Expects to output 1 for real, 0 for fake
- **Generator**: Tries to trick discriminator (outputs should approach 1)
- Binary cross-entropy with logits for numerical stability

## Theory

### GAN Training Loop
1. **Discriminator Step**: 
   - Get real images from dataset
   - Generate fake images with generator
   - Classify both as real/fake
   - Update discriminator to improve classification

2. **Generator Step**:
   - Generate fake images
   - Try to fool discriminator
   - Update generator to improve fake image quality

### Key Techniques Used
- **Batch Normalization**: Stabilizes training
- **Leaky ReLU**: Prevents dead neurons in discriminator
- **Strided Convolutions**: Downsampling instead of pooling
- **Transposed Convolutions**: Upsampling in generator
- **Label Smoothing**: Prevents discriminator from becoming too confident
- **Variable Scoping**: Enables weight reuse for inference

## Expected Results

After training:
- **MNIST**: Generator learns to create realistic handwritten digits
- **CelebA**: Generator learns to synthesize plausible face images
- **Loss Curves**: Should stabilize with generator and discriminator losses balancing

## Notes

- GPU recommended for training (significantly faster)
- First time dataset downloads are large (~2GB for CelebA)
- Training can take hours depending on hardware
- Generated images quality improves over epochs
- Early stopping or checkpointing recommended to preserve best models

## References

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [TensorFlow GAN Documentation](https://www.tensorflow.org/tutorials/generative/dcgan)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)

## License

This is an educational project part of the Udacity Deep Learning Nanodegree program.
