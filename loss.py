import tensorflow as tf
import tensorflow.backend as K

def wasserstein_loss(y_true, y_pred):
    """Not really Wasserstein loss but almost!"""
    return K.mean(y_true * y_pred)
