# ViT from scratch

This project implements a Vision Transformer (ViT) model from scratch using PyTorch. The model is designed to process images by treating them as sequences of patches, similar to how a standard Transformer handles sequences of words.

## Project Structure

-   `vit.py`: Contains the core Vision Transformer model architecture.
-   `image_processing.py`: Provides utility functions for image preprocessing.
-   `README.md`: This file, providing an overview of the project and its components.

## Model Architecture (`vit.py`)

The `vit.py` file defines the core components of the Vision Transformer:

-   **`ViTConfig`**: Manages the hyperparameters and configuration settings for the ViT model, including image size, patch size, hidden dimensions, number of attention heads, and dropout probabilities.
-   **`ViTPatchEmbeddings`**: Converts input images into a sequence of flattened 2D patches and projects them into a higher-dimensional space using a convolutional layer.
-   **`ViTEmbeddings`**: Combines the patch embeddings with a learnable `[CLS]` token (used for classification) and adds positional embeddings to encode spatial information. Dropout is applied to these combined embeddings.
-   **`ViTSelfAttention`**: Implements the multi-head self-attention mechanism, which allows the model to weigh the importance of different parts of the input sequence (patches) when processing each patch.
-   **`ViTAttention`**: Encapsulates the self-attention mechanism and its output processing.
-   **`ViTIntermediate`**: A feed-forward network layer with a GELU activation function, applied after the attention mechanism within each transformer block.
-   **`ViTOutput`**: Another feed-forward network layer that includes a residual connection, completing the feed-forward part of a transformer block.
-   **`ViTLayer`**: Represents a single Transformer encoder block, comprising a multi-head self-attention module, two layer normalization layers, and a feed-forward network.
-   **`ViTEncoder`**: Stacks multiple `ViTLayer` blocks to form the main encoder part of the Vision Transformer, processing the sequence of patch embeddings.
-   **`VisionTransformer`**: The main model class that integrates the `ViTEmbeddings` and `ViTEncoder` to form the complete Vision Transformer. It takes raw pixel values as input and outputs the representation of the `[CLS]` token, which can then be used for downstream tasks like image classification.

## Image Preprocessing (`image_processing.py`)

The `image_processing.py` file contains the `preprocess_image` function, which is responsible for preparing raw images for input into the Vision Transformer. This function performs the following steps:

-   **Resizing**: Resizes the input image to the specified `image_size` (defaulting to 224x224 pixels).
-   **ToTensor**: Converts the image from a PIL Image or NumPy array to a PyTorch `Tensor`.
-   **Normalization**: Normalizes the pixel values using predefined mean and standard deviation values, which are typical for models pre-trained on ImageNet.
-   **Batch Dimension**: Adds an extra dimension to the tensor to represent the batch size, making it suitable for model input.

## How to Run

To run the `vit.py` script and test the model's forward pass with dummy input:

```bash
uv run vit.py
```
This will execute the example in the `if __name__ == '__main__':` block, demonstrating the shape of the CLS token output.