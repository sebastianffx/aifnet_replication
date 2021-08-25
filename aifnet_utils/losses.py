import tensorflow_probability as tfp
import tensorflow as tf

# Custom Loss Function
def MaxCorrelation(y_true,y_pred):
    """
    Goal is to maximize correlation between y_pred, y_true. Same as minimizing the negative.
    """
    return -tf.math.abs(tfp.stats.correlation(y_pred,y_true, sample_axis=None, event_axis=None))

