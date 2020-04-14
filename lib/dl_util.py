"""
Frequently used layer combinations
"""
# imports for building the network
import tensorflow as tf
import keras
import keras.backend as kb
from tensorflow import reduce_sum
from keras.backend import pow
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten, LeakyReLU
from keras.losses import binary_crossentropy
import configparser


def bn_activation(x, activation="relu"):
    """
    batch normalization layer with an optional activation layer
    Args:
        x:
        activation:

    Returns:

    """
    x = keras.layers.BatchNormalization()(x)
    if activation != '':
        x = keras.layers.Activation(activation)(x)

    return x


def conv_bn_block(x, filters, kernel_size=3, padding='same', strides=1, activation="relu"):
    """
    Convolution layer using the batch normalization layer
    Args:
        x:
        filters:
        kernel_size:
        padding:
        strides:
        activation:

    Returns:

    """
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    bn = bn_activation(conv, activation=activation)

    return bn


def residual_block(x, filters, kernel_size=3, padding='same', strides=1, activation='relu'):
    """

    Args:
        x:
        filters:
        kernel_size:
        padding:
        strides:
        activation:

    Returns:

    """
    res = conv_bn_block(x, filters, kernel_size, padding=padding, strides=strides, activation=activation)
    res = conv_bn_block(res, filters, kernel_size, padding=padding, strides=strides, activation=activation)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_activation(shortcut, activation="")
    output = Add()([shortcut, res])

    return output


def ini_to_dict(ini_path):
    """
    load all info from ini as str
    Args:
        ini_path:

    Returns:

    """
    config = configparser.ConfigParser()
    config.read(ini_path)

    config_dict = {}
    for section in config.sections():
        sect_dict = {}
        for option in config.options(section):
            sect_dict[option] = config.get(section, option)
        config_dict[section] = sect_dict

    return config_dict








