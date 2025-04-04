from tensorflow.keras import layers, models
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
# Constants for image size and batch size
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224  # Input size for DenseNet121
BATCH_SIZE = 8

def build_densenet121_feature_extractor():
    # Load the DenseNet121 model pre-trained on ImageNet, excluding the top layer (classification layer)
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    # Freeze the base model layers (optional, can unfreeze if fine-tuning is required)
    base_model.trainable = False

    # Create a model that outputs the feature map after Global Average Pooling
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Convert 2D feature map to a 1D feature vector
    ])

    return model


class PositionalEmbedding(layers.Layer):
    """
    Custom layer for adding positional embeddings to input sequences.
    Positional embeddings are used to retain information about the order of elements in a sequence.
    """

    def __init__(self, sequence_length, output_dim, **kwargs):
        """
        Initializes the positional embedding layer.

        Args:
            sequence_length (int): The length of the input sequence.
            output_dim (int): The dimensionality of the embeddings.
            **kwargs: Additional arguments for the base Layer class.
        """
        super().__init__(**kwargs)

        # Embedding layer for positional information, where `input_dim` is the sequence length
        # and `output_dim` is the dimensionality of the embedding space.
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

        # Store sequence length and output dimension for reference
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        """
        Adds positional embeddings to the input sequence.

        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, frames, num_features)`.

        Returns:
            Tensor: Input tensor with added positional embeddings of the same shape.
        """

        # Ensure inputs are in the correct data type as per the layer's computation requirements
        inputs = tf.cast(inputs, self.compute_dtype)

        # Dynamically determine the sequence length of the input (second dimension)
        length = tf.shape(inputs)[1]

        # Create a range tensor representing positional indices (0, 1, ..., length - 1)
        positions = tf.range(start=0, limit=length, delta=1)

        # Get positional embeddings for each index in the range `positions`
        embedded_positions = self.position_embeddings(positions)

        # Add the positional embeddings to the input tensor to incorporate position information
        return inputs + embedded_positions

class TransformerEncoder(layers.Layer):
    """
    Custom Transformer Encoder layer.
    This layer includes multi-head self-attention and feed-forward network blocks
    with residual connections and layer normalization, commonly used in transformer architectures.
    """

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        """
        Initializes the Transformer encoder layer.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            dense_dim (int): Dimensionality of the feed-forward network.
            num_heads (int): Number of attention heads.
            **kwargs: Additional keyword arguments for the base Layer class.
        """
        super().__init__(**kwargs)

        # Store parameters
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Multi-head self-attention layer with dropout for regularization
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )

        # Feed-forward network with two dense layers and GELU activation
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation=keras.activations.gelu),
            layers.Dense(embed_dim),
        ])

        # Layer normalization layers to stabilize training and enhance performance
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        """
        Forward pass for the Transformer Encoder.

        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, sequence_length, embed_dim)`.
            mask (Tensor, optional): Optional mask tensor for attention mechanism.

        Returns:
            Tensor: Output tensor of shape `(batch_size, sequence_length, embed_dim)`.
        """

        # Self-attention mechanism: inputs attend to themselves
        attention_output = self.attention(inputs, inputs, attention_mask=mask)

        # Residual connection and layer normalization after self-attention
        proj_input = self.layernorm_1(inputs + attention_output)

        # Feed-forward network applied to the normalized input
        proj_output = self.dense_proj(proj_input)

        # Second residual connection and layer normalization after the feed-forward network
        return self.layernorm_2(proj_input + proj_output)



# def get_compiled_model(shape):
#     """
#     Builds and compiles a Transformer-based model for sequence classification.

#     Args:
#         shape (tuple): Shape of the input tensor, typically (sequence_length, embed_dim).

#     Returns:
#         keras.Model: Compiled Keras model ready for training.
#     """

#     # Define model parameters
#     sequence_length = MAX_SEQ_LENGTH  # Length of the input sequence
#     embed_dim = NUM_FEATURES          # Dimensionality of each feature in the input
#     dense_dim = 512                   # Dimensionality of the dense layer in TransformerEncoder
#     num_heads = 2                     # Number of attention heads in TransformerEncoder
#     classes = 5                       # Number of output classes for classification

#     # Define model input
#     inputs = keras.Input(shape=shape)

#     # Add positional embeddings to the input sequence
#     x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)

#     # Pass the input through a Transformer encoder layer
#     x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)

#     # Apply global max pooling to reduce the sequence dimension
#     x = layers.GlobalMaxPooling1D()(x)

#     # Add dropout for regularization to prevent overfitting
#     x = layers.Dropout(0.5)(x)

#     # Output dense layer with softmax activation for multi-class classification
#     outputs = layers.Dense(classes, activation="softmax")(x)

#     # Create the model by specifying the inputs and outputs
#     model = keras.Model(inputs, outputs)

#     # Compile the model with Adam optimizer and sparse categorical cross-entropy loss
#     model.compile(
#         optimizer="adam",
#         loss="SparseCategoricalCrossentropy",  # Use sparse categorical cross-entropy for integer labels
#         metrics=["accuracy"],  # Track accuracy during training
#     )

#     return model
def get_compiled_model(input_shape, sequence_length, embed_dim, num_classes, dense_dim, num_heads):
    inputs = keras.Input(shape=input_shape)
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="SparseCategoricalCrossentropy",
        metrics=["accuracy"],
    )

    return model