"""
class HomographyNet
"""
import pandas as pd
from keras.models import Model
from keras.layers import Input, MaxPooling2D, Flatten, LeakyReLU, Dense, Reshape, Dropout, Activation
from keras import optimizers
import tensorflow as tf

from lib import dl_util


class HomographyNet:
    def __init__(self, input_size=(448, 448, 2), layer_config="default_net_config.ini"):
        self.input_size = input_size
        self.layer_config = dl_util.ini_to_dict(layer_config)
        self.layer_names = list(self.layer_config.keys())
        self.model = None

    def build_model(self, opt="adam", loss=tf.keras.losses.MSE, metrics=tf.keras.losses.MSE):
        inputs = Input(self.input_size)

        x = dl_util.conv_bn_block(inputs, filters=int(self.layer_config["conv1"]["nb_filter"]),
                                  kernel_size=int(self.layer_config["conv1"]["k_size"]),
                                  padding=self.layer_config["conv1"]["padding"],
                                  strides=int(self.layer_config["conv1"]["strides"]),
                                  activation=self.layer_config["conv1"]["activation"]
                                  )

        for lay_name in self.layer_names[1:]:
            if lay_name[:-1] == 'conv':
                x = dl_util.conv_bn_block(x, filters=int(self.layer_config[lay_name]["nb_filter"]),
                                          kernel_size=int(self.layer_config[lay_name]["k_size"]),
                                          padding=self.layer_config[lay_name]["padding"],
                                          strides=int(self.layer_config[lay_name]["strides"]),
                                          activation=self.layer_config[lay_name]["activation"]
                                          )
            elif lay_name[:-1] == 'mp':
                x = MaxPooling2D((int(self.layer_config[lay_name]["k_size"]),
                                  int(self.layer_config[lay_name]["k_size"])),
                                 strides=int(self.layer_config[lay_name]["strides"])
                                 )(x)

        x = Flatten()(x)
        x = Dropout(rate=float(self.layer_config["dropout1"]["dropout"]))(x)
        x = Dense(int(self.layer_config["fc1"]["nb_node"]))(x)
        x = Activation(activation=self.layer_config["fc1"]["activation"])(x)

        outputs = Dense(int(self.layer_config["fc2"]["nb_node"]))(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mean_squared_error"])

        return model


def main():
    homo_net = HomographyNet()

    model = homo_net.build_model()

    model.summary()


if __name__ == '__main__':
    main()

