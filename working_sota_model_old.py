import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, Concatenate, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

def attention_gate(g, s, num_filters):
    """Attention gate ensuring 5D tensors"""
    g_conv = Conv3D(num_filters, 1, padding='same')(g)
    s_conv = Conv3D(num_filters, 1, padding='same')(s)
    f = tf.keras.layers.Add()([g_conv, s_conv])
    f = BatchNormalization()(f)
    f = LeakyReLU(0.01)(f)
    f = Conv3D(1, 1, padding='same')(f)
    f = tf.keras.layers.Activation('sigmoid')(f)
    output = tf.keras.layers.Multiply()([s, f])
    if len(output.shape) != 5:
        raise ValueError(f"Attention gate output shape {output.shape} is not 5D")
    return output

def build_sota_model(input_shape=(192, 224, 176, 1), base_filters=32):
    """Build 3D Attention U-Net with full complexity"""
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv3D(base_filters, 3, padding='same')(inputs)
    c1 = LeakyReLU(0.01)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(base_filters, 3, padding='same')(c1)
    c1 = LeakyReLU(0.01)(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling3D(2)(c1)

    c2 = Conv3D(base_filters*2, 3, padding='same')(p1)
    c2 = LeakyReLU(0.01)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(base_filters*2, 3, padding='same')(c2)
    c2 = LeakyReLU(0.01)(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling3D(2)(c2)

    c3 = Conv3D(base_filters*4, 3, padding='same')(p2)
    c3 = LeakyReLU(0.01)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(base_filters*4, 3, padding='same')(c3)
    c3 = LeakyReLU(0.01)(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling3D(2)(c3)

    c4 = Conv3D(base_filters*8, 3, padding='same')(p3)
    c4 = LeakyReLU(0.01)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv3D(base_filters*8, 3, padding='same')(c4)
    c4 = LeakyReLU(0.01)(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling3D(2)(c4)

    # Bottleneck
    c5 = Conv3D(base_filters*16, 3, padding='same')(p4)
    c5 = LeakyReLU(0.01)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv3D(base_filters*16, 3, padding='same')(c5)
    c5 = LeakyReLU(0.01)(c5)
    c5 = BatchNormalization()(c5)

    # Decoder
    u6 = Conv3DTranspose(base_filters*8, 2, strides=2, padding='same')(c5)
    a6 = attention_gate(c4, u6, base_filters*8)
    u6 = Concatenate()([u6, a6])
    c6 = Conv3D(base_filters*8, 3, padding='same')(u6)
    c6 = LeakyReLU(0.01)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv3D(base_filters*8, 3, padding='same')(c6)
    c6 = LeakyReLU(0.01)(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv3DTranspose(base_filters*4, 2, strides=2, padding='same')(c6)
    a7 = attention_gate(c3, u7, base_filters*4)
    u7 = Concatenate()([u7, a7])
    c7 = Conv3D(base_filters*4, 3, padding='same')(u7)
    c7 = LeakyReLU(0.01)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv3D(base_filters*4, 3, padding='same')(c7)
    c7 = LeakyReLU(0.01)(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv3DTranspose(base_filters*2, 2, strides=2, padding='same')(c7)
    a8 = attention_gate(c2, u8, base_filters*2)
    u8 = Concatenate()([u8, a8])
    c8 = Conv3D(base_filters*2, 3, padding='same')(u8)
    c8 = LeakyReLU(0.01)(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv3D(base_filters*2, 3, padding='same')(c8)
    c8 = LeakyReLU(0.01)(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv3DTranspose(base_filters, 2, strides=2, padding='same')(c8)
    a9 = attention_gate(c1, u9, base_filters)
    u9 = Concatenate()([u9, a9])
    c9 = Conv3D(base_filters, 3, padding='same')(u9)
    c9 = LeakyReLU(0.01)(c9)
    c9 = BatchNormalization()(c9)
    c9 = Conv3D(base_filters, 3, padding='same')(c9)
    c9 = LeakyReLU(0.01)(c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv3D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs, outputs, name="sota_attention_unet")
    return model
