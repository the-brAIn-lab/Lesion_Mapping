import tensorflow as tf
from working_sota_model import build_sota_model

# Build model
model = build_sota_model(input_shape=(192, 224, 176, 1), base_filters=32)
model.summary()

# Test forward pass
test_input = tf.random.normal((1, 192, 224, 176, 1))
print("Input shape:", test_input.shape)
output = model(test_input)
print("Output shape:", output.shape)

# Test backpropagation
with tf.GradientTape() as tape:
    pred = model(test_input, training=True)
    loss = tf.reduce_mean(pred)  # Dummy loss
grads = tape.gradient(loss, model.trainable_variables)
print("Gradient shapes:", [g.shape if g is not None else None for g in grads])
