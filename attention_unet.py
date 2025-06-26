"""
attention_unet.py - State-of-the-art Attention U-Net for stroke lesion segmentation
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class AttentionGate(layers.Layer):
    """Attention gate for focusing on relevant features"""
    def __init__(self, num_filters, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        
    def build(self, input_shape):
        # Define layers
        self.W_g = layers.Conv3D(self.num_filters, kernel_size=1, padding='same')
        self.W_x = layers.Conv3D(self.num_filters, kernel_size=1, padding='same')
        self.psi = layers.Conv3D(1, kernel_size=1, padding='same')
        self.relu = layers.ReLU()
        self.sigmoid = layers.Activation('sigmoid')
        self.bn_g = layers.BatchNormalization()
        self.bn_x = layers.BatchNormalization()
        
    def call(self, x, g, training=None):
        """
        x: input features (from skip connection)
        g: gating signal (from lower layer)
        """
        # Apply convolutions
        W_g_out = self.W_g(g)
        W_g_out = self.bn_g(W_g_out, training=training)
        
        W_x_out = self.W_x(x)
        W_x_out = self.bn_x(W_x_out, training=training)
        
        # Add and apply ReLU
        combined = self.relu(W_g_out + W_x_out)
        
        # Apply psi and sigmoid to get attention coefficients
        attention_coefficients = self.sigmoid(self.psi(combined))
        
        # Apply attention to input features
        return x * attention_coefficients


class ConvBlock(layers.Layer):
    """Convolutional block with BatchNorm and LeakyReLU"""
    def __init__(self, num_filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv1 = layers.Conv3D(
            self.num_filters, 
            self.kernel_size, 
            padding='same',
            kernel_initializer='he_normal'
        )
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.LeakyReLU(alpha=0.1)
        
        self.conv2 = layers.Conv3D(
            self.num_filters, 
            self.kernel_size, 
            padding='same',
            kernel_initializer='he_normal'
        )
        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.LeakyReLU(alpha=0.1)
        
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation2(x)
        
        return x


class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, num_filters, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.reduction = reduction
        
    def build(self, input_shape):
        self.gap = layers.GlobalAveragePooling3D()
        self.fc1 = layers.Dense(self.num_filters // self.reduction, activation='relu')
        self.fc2 = layers.Dense(self.num_filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, 1, self.num_filters))
        
    def call(self, x):
        # Squeeze
        squeeze = self.gap(x)
        
        # Excitation
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        excitation = self.reshape(excitation)
        
        # Scale
        return x * excitation


def build_attention_unet(
    input_shape=(192, 224, 176, 1),
    num_classes=1,
    filters=[32, 64, 128, 256, 512],
    dropout_rate=0.3,
    use_se_blocks=True
):
    """
    Build Attention U-Net with optional SE blocks
    
    Args:
        input_shape: Input volume shape
        num_classes: Number of output classes
        filters: List of filter numbers for each level
        dropout_rate: Dropout rate for regularization
        use_se_blocks: Whether to use Squeeze-and-Excitation blocks
    """
    inputs = layers.Input(shape=input_shape)
    
    # Store skip connections
    skip_connections = []
    
    # Encoder path
    x = inputs
    for i, f in enumerate(filters[:-1]):
        # Convolutional block
        x = ConvBlock(f, name=f'encoder_block_{i}')(x)
        
        # Add SE block if enabled
        if use_se_blocks:
            x = SEBlock(f, name=f'se_block_{i}')(x)
        
        # Store skip connection
        skip_connections.append(x)
        
        # Downsample
        x = layers.MaxPooling3D(pool_size=2, name=f'pool_{i}')(x)
        
        # Add dropout for middle layers
        if i > 0 and i < len(filters) - 2:
            x = layers.SpatialDropout3D(dropout_rate)(x)
    
    # Bottleneck
    x = ConvBlock(filters[-1], name='bottleneck')(x)
    if use_se_blocks:
        x = SEBlock(filters[-1], name='se_bottleneck')(x)
    x = layers.SpatialDropout3D(dropout_rate)(x)
    
    # Decoder path with attention gates
    for i in reversed(range(len(filters[:-1]))):
        # Upsample
        x = layers.Conv3DTranspose(
            filters[i], 
            kernel_size=2, 
            strides=2, 
            padding='same',
            name=f'upsample_{i}'
        )(x)
        
        # Apply attention gate
        skip = skip_connections[i]
        attention = AttentionGate(filters[i], name=f'attention_{i}')(skip, x)
        
        # Concatenate
        x = layers.Concatenate()([x, attention])
        
        # Convolutional block
        x = ConvBlock(filters[i], name=f'decoder_block_{i}')(x)
        
        # Add dropout for middle layers
        if i > 0:
            x = layers.SpatialDropout3D(dropout_rate/2)(x)
    
    # Output layer
    outputs = layers.Conv3D(
        num_classes,
        kernel_size=1,
        activation='sigmoid',
        name='output'
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='attention_unet')
    
    return model


def build_deep_supervision_model(base_model, weights=[1.0, 0.5, 0.25, 0.125]):
    """
    Add deep supervision to the model
    
    Args:
        base_model: Base Attention U-Net model
        weights: Weights for each supervision level
    """
    inputs = base_model.input
    
    # Get intermediate outputs from decoder blocks
    supervision_outputs = []
    
    # Get the main output
    main_output = base_model.output
    supervision_outputs.append(main_output)
    
    # Get intermediate decoder outputs and upsample them
    for i, w in enumerate(weights[1:]):
        # Get intermediate layer output
        layer_name = f'decoder_block_{len(weights)-2-i}'
        intermediate = base_model.get_layer(layer_name).output
        
        # Add 1x1 conv for prediction
        pred = layers.Conv3D(1, 1, activation='sigmoid', 
                           name=f'deep_supervision_{i}')(intermediate)
        
        # Upsample to match input size
        for _ in range(len(weights)-1-i):
            pred = layers.Conv3DTranspose(1, 2, strides=2, padding='same',
                                        name=f'deep_upsample_{i}_{_}')(pred)
        
        supervision_outputs.append(pred)
    
    # Create model with multiple outputs
    model = Model(inputs=inputs, outputs=supervision_outputs, 
                  name='attention_unet_deep_supervision')
    
    return model, weights


class DynamicResizeLayer(layers.Layer):
    """Custom layer to handle dynamic input sizes"""
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape
        
    def call(self, inputs):
        # Use TensorFlow's resize_volumes (trilinear interpolation)
        # First, transpose to (batch, depth, height, width, channels)
        x = tf.transpose(inputs, [0, 3, 1, 2, 4])
        
        # Resize
        x = tf.image.resize(x, self.target_shape[1:3], method='bilinear')
        
        # Resize depth dimension
        x = tf.transpose(x, [0, 2, 3, 1, 4])
        x = tf.image.resize(x, [self.target_shape[2], self.target_shape[0]], method='bilinear')
        
        # Transpose back
        x = tf.transpose(x, [0, 3, 1, 2, 4])
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'target_shape': self.target_shape})
        return config


# Test the model
if __name__ == "__main__":
    # Build model
    model = build_attention_unet(
        input_shape=(192, 224, 176, 1),
        filters=[32, 64, 128, 256, 512],
        use_se_blocks=True
    )
    
    # Print model summary
    model.summary()
    
    # Test with deep supervision
    deep_model, weights = build_deep_supervision_model(model)
    print(f"\nDeep supervision model created with {len(deep_model.outputs)} outputs")
