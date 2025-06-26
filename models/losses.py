"""
losses.py - Custom loss functions for stroke lesion segmentation
Handles extreme class imbalance (lesions are 0.02%-1.5% of brain volume)
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for binary segmentation
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice loss for gradient descent
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def generalized_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Generalized Dice coefficient that weights classes by inverse volume
    Better for extreme class imbalance
    """
    # Calculate class weights (inverse of squared volume)
    w0 = 1.0 / (K.sum(1 - y_true) ** 2 + smooth)
    w1 = 1.0 / (K.sum(y_true) ** 2 + smooth)
    
    # Normalize weights
    w_sum = w0 + w1
    w0 = w0 / w_sum
    w1 = w1 / w_sum
    
    # Calculate weighted intersection and union
    intersection = w1 * K.sum(y_true * y_pred) + w0 * K.sum((1 - y_true) * (1 - y_pred))
    union = w1 * K.sum(y_true + y_pred) + w0 * K.sum(2 - y_true - y_pred)
    
    return (2. * intersection + smooth) / (union + smooth)


def generalized_dice_loss(y_true, y_pred):
    """
    Generalized Dice loss
    """
    return 1.0 - generalized_dice_coefficient(y_true, y_pred)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for addressing class imbalance
    alpha: Weight for positive class
    gamma: Focusing parameter (higher = more focus on hard examples)
    """
    # Clip predictions to prevent log(0)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    # Calculate focal loss
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
    
    focal_loss = -alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t)
    
    return K.mean(focal_loss)


def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    """
    Focal Tversky loss - combination of Tversky index and focal loss
    Better for small lesion detection
    alpha: Weight for false negatives (higher = more penalty for missing lesions)
    beta: Weight for false positives
    gamma: Focal parameter
    """
    # Flatten tensors
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # True positives, false positives, false negatives
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    
    # Tversky index
    tversky = (tp + K.epsilon()) / (tp + alpha * fn + beta * fp + K.epsilon())
    
    # Focal Tversky loss
    return K.pow(1 - tversky, gamma)


def boundary_loss(y_true, y_pred, theta0=3, theta=5):
    """
    Boundary loss for better lesion boundary delineation
    Uses distance transform of ground truth
    """
    # Get boundaries using morphological operations
    kernel = tf.ones((3, 3, 3, 1, 1))
    
    # Dilate and erode to get boundaries
    y_true_dilated = tf.nn.dilation2d(
        tf.expand_dims(y_true, 0),
        filters=kernel,
        strides=[1, 1, 1, 1, 1],
        padding='SAME',
        data_format='NDHWC',
        dilations=[1, 1, 1, 1, 1]
    )
    
    y_true_eroded = tf.nn.erosion2d(
        tf.expand_dims(y_true, 0),
        filters=kernel,
        strides=[1, 1, 1, 1, 1],
        padding='SAME',
        data_format='NDHWC',
        dilations=[1, 1, 1, 1, 1]
    )
    
    # Boundary = dilated - eroded
    boundary = tf.squeeze(y_true_dilated - y_true_eroded, 0)
    
    # Extended boundary
    extended_boundary = tf.nn.dilation2d(
        tf.expand_dims(boundary, 0),
        filters=kernel,
        strides=[1, 1, 1, 1, 1],
        padding='SAME',
        data_format='NDHWC',
        dilations=[1, 1, 1, 1, 1]
    )
    extended_boundary = tf.squeeze(extended_boundary, 0)
    
    # Compute boundary loss
    P_boundary = y_pred * boundary
    P_extended = y_pred * extended_boundary
    
    # Smooth approximation of L1 loss
    smooth_l1 = K.mean(tf.where(
        P_boundary < theta0,
        0.5 * P_boundary * P_boundary / theta0,
        P_boundary - 0.5 * theta0
    ))
    
    loss = theta * smooth_l1 + K.mean(P_extended)
    
    return loss


def combined_loss(
    dice_weight=1.0,
    focal_weight=1.0,
    boundary_weight=0.5,
    use_generalized_dice=True,
    focal_alpha=0.25,
    focal_gamma=3.0
):
    """
    Combined loss function for stroke segmentation
    """
    def loss(y_true, y_pred):
        # Choose dice variant
        if use_generalized_dice:
            d_loss = generalized_dice_loss(y_true, y_pred)
        else:
            d_loss = dice_loss(y_true, y_pred)
        
        # Focal loss
        f_loss = focal_loss(y_true, y_pred, alpha=focal_alpha, gamma=focal_gamma)
        
        # Boundary loss
        b_loss = boundary_loss(y_true, y_pred)
        
        # Combine losses
        total_loss = (dice_weight * d_loss + 
                     focal_weight * f_loss + 
                     boundary_weight * b_loss)
        
        return total_loss
    
    return loss


def deep_supervision_loss(weights=[1.0, 0.5, 0.25, 0.125]):
    """
    Loss function for deep supervision
    weights: List of weights for each supervision level
    """
    base_loss = combined_loss()
    
    def loss(y_true, y_pred_list):
        total_loss = 0
        
        for i, (pred, weight) in enumerate(zip(y_pred_list, weights)):
            total_loss += weight * base_loss(y_true, pred)
        
        return total_loss / sum(weights)
    
    return loss


class AdaptiveLossWeight(tf.keras.callbacks.Callback):
    """
    Callback to adaptively adjust loss weights during training
    based on performance
    """
    def __init__(self, loss_fn, patience=5):
        super().__init__()
        self.loss_fn = loss_fn
        self.patience = patience
        self.wait = 0
        self.best_dice = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_dice = logs.get('dice_coefficient', 0)
        
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.wait = 0
        else:
            self.wait += 1
            
        # Adjust weights if no improvement
        if self.wait >= self.patience:
            # Increase focal weight to focus more on hard examples
            if hasattr(self.loss_fn, 'focal_weight'):
                self.loss_fn.focal_weight *= 1.1
                print(f"\nIncreasing focal weight to {self.loss_fn.focal_weight:.3f}")
            self.wait = 0


# Metrics for evaluation
def sensitivity(y_true, y_pred):
    """Recall/Sensitivity - important for not missing lesions"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    """Specificity - important for not over-segmenting"""
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def precision(y_true, y_pred):
    """Precision - ratio of true positives to predicted positives"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


if __name__ == "__main__":
    # Test losses with synthetic data
    import numpy as np
    
    # Create synthetic data (small lesion)
    y_true = np.zeros((1, 192, 224, 176, 1), dtype=np.float32)
    y_true[0, 90:100, 110:120, 85:95, 0] = 1.0  # Small lesion
    
    y_pred = np.random.random((1, 192, 224, 176, 1)).astype(np.float32) * 0.5
    y_pred[0, 88:102, 108:122, 83:97, 0] = 0.8  # Slightly misaligned prediction
    
    # Convert to tensors
    y_true_tensor = tf.constant(y_true)
    y_pred_tensor = tf.constant(y_pred)
    
    # Test different losses
    print("Testing loss functions:")
    print(f"Dice Loss: {dice_loss(y_true_tensor, y_pred_tensor).numpy():.4f}")
    print(f"Generalized Dice Loss: {generalized_dice_loss(y_true_tensor, y_pred_tensor).numpy():.4f}")
    print(f"Focal Loss: {focal_loss(y_true_tensor, y_pred_tensor).numpy():.4f}")
    print(f"Focal Tversky Loss: {focal_tversky_loss(y_true_tensor, y_pred_tensor).numpy():.4f}")
    
    # Test combined loss
    combined = combined_loss()
    print(f"Combined Loss: {combined(y_true_tensor, y_pred_tensor).numpy():.4f}")
    
    # Test metrics
    print(f"\nMetrics:")
    print(f"Dice Coefficient: {dice_coefficient(y_true_tensor, y_pred_tensor).numpy():.4f}")
    print(f"Sensitivity: {sensitivity(y_true_tensor, y_pred_tensor).numpy():.4f}")
    print(f"Specificity: {specificity(y_true_tensor, y_pred_tensor).numpy():.4f}")
    print(f"Precision: {precision(y_true_tensor, y_pred_tensor).numpy():.4f}")
