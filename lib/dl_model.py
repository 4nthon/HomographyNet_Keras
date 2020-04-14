"""
Deep learning model
"""
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, LeakyReLU, Dense, Reshape
import tensorflow as tf

from . import dl_util


class YoloV1:
    def __init__(self, input_size=(448, 448, 3), nb_box_per_cell=2, nb_class=20, cell_grid=(7, 7), layer_config=None):
        self.input_size = input_size
        self.nb_class = nb_class
        self.cell_grid = cell_grid
        self.nb_box_per_cell = nb_box_per_cell
        if layer_config is not None:
            self.layers = pd.read_csv(layer_config)
        else:
            layer_info = [["conv", 64, 7, 2, "same"],
                          ["mp", 0, 2, 2, "same"],
                          ["conv", 192, 3, 1, "same"],
                          ["mp", 0, 2, 2, "same"],
                          ["conv", 128, 1, 1, "same"],
                          ["conv", 256, 3, 1, "same"],
                          ["conv", 256, 1, 1, "same"],
                          ["conv", 512, 3, 1, "same"],
                          ["mp", 0, 2, 2, "same"],
                          ["conv", 256, 1, 1, "same"],
                          ["conv", 512, 3, 1, "same"],
                          ["conv", 256, 1, 1, "same"],
                          ["conv", 512, 3, 1, "same"],
                          ["conv", 256, 1, 1, "same"],
                          ["conv", 512, 3, 1, "same"],
                          ["conv", 256, 1, 1, "same"],
                          ["conv", 512, 3, 1, "same"],
                          ["conv", 512, 1, 1, "same"],
                          ["conv", 1024, 3, 1, "same"],
                          ["mp", 0, 2, 2, "same"],
                          ["conv", 512, 1, 1, "same"],
                          ["conv", 1024, 3, 1, "same"],
                          ["conv", 512, 1, 1, "same"],
                          ["conv", 1024, 3, 1, "same"],
                          ["conv", 1024, 3, 1, "same"],
                          ["conv", 1024, 3, 2, "same"],
                          ["conv", 1024, 3, 1, "same"],
                          ["conv", 1024, 3, 1, "same"],
                          ["fc", 4096, 0, 0, 0],
                          ]
            self.layers = pd.DataFrame(layer_info, columns=["layer", "nb_filter", "k_size", "strides", "padding"])

    def make_model(self):
        nb_layer = self.layers.shape[0]

        inputs = Input(self.input_size)
        x = dl_util.yolo_conv_block_v1(inputs, int(self.layers.loc[0]['nb_filter']),
                                       kernel_size=int(self.layers.loc[0]['k_size']),
                                       strides=int(self.layers.loc[0]['strides']),
                                       padding=self.layers.loc[0]['padding']
                                       )

        for idx in range(1, nb_layer - 1):
            if self.layers.loc[idx]['layer'] == 'conv':
                x = dl_util.yolo_conv_block_v1(x, int(self.layers.loc[idx]['nb_filter']),
                                               kernel_size=int(self.layers.loc[idx]['k_size']),
                                               strides=int(self.layers.loc[idx]['strides']),
                                               padding=self.layers.loc[idx]['padding']
                                               )
            elif self.layers.loc[idx]['layer'] == 'mp':
                x = MaxPooling2D((int(self.layers.loc[idx]['k_size']), int(self.layers.loc[idx]['k_size'])),
                                 strides=int(self.layers.loc[idx]['strides'])
                                 )(x)

        x = Flatten()(x)
        x = Dense(int(self.layers.loc[nb_layer-1]['nb_filter']))(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Dense(self.cell_grid[0] * self.cell_grid[1] * (5 * self.nb_box_per_cell + self.nb_class))(x)

        outputs = Reshape([self.cell_grid[0], self.cell_grid[1], 5 * self.nb_box_per_cell + self.nb_class])(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss=self.yolo_loss_v1_2box, metrics=['accuracy'])

        return model

    def yolo_loss_v1_2box(self, y_true, y_pred):
        """
        This loss function is only for the yolo who predict 2 boxes in one cell
        Args:
            y_true:
            y_pred:

        Returns:

        """
        batch_size = tf.shape(y_pred)[0]
        xy1_true, wh1_true, p1_true, xy2_true, wh2_true, p2_true, c_true = tf.split(y_true,
                                                                                    (2, 2, 1, 2, 2, 1, self.nb_class),
                                                                                    axis=-1)
        xy1_pred, wh1_pred, p1_pred, xy2_pred, wh2_pred, p2_pred, c_pred = tf.split(y_pred,
                                                                                    (2, 2, 1, 2, 2, 1, self.nb_class),
                                                                                    axis=-1)

        loss_xy_1 = tf.reduce_sum(tf.multiply(tf.square(tf.add(xy1_true, -xy1_pred)), p1_true))
        loss_wh_1 = tf.reduce_sum(tf.multiply(tf.square(tf.add(tf.sqrt(wh1_true), -tf.sqrt(wh1_pred))), p1_true),
                                  axis=0)
        loss_pro_1 = tf.reduce_sum(tf.multiply(tf.square(tf.add(p1_true, -p1_pred)), p1_true))

        loss_xy_2 = tf.reduce_sum(tf.multiply(tf.square(tf.add(xy2_true, -xy2_pred)), p2_true))
        loss_wh_2 = tf.reduce_sum(tf.multiply(tf.square(tf.add(tf.sqrt(wh2_true), -tf.sqrt(wh2_pred))), p2_true),
                                  axis=0)
        loss_pro_2 = tf.reduce_sum(tf.multiply(tf.square(tf.add(p2_true, -p2_pred)), p2_true))

        loss_cls_obj = tf.reduce_sum(tf.multiply(tf.square(tf.add(c_true, - c_pred)), c_true))
        loss_cls_noobj = tf.reduce_sum(tf.multiply(tf.square(tf.add(c_true, - c_pred)), tf.add(1, -c_true)))

        loss = 5 * (loss_xy_1 + loss_xy_2 + loss_wh_1 + loss_wh_2) + loss_cls_obj + 0.5 * loss_cls_noobj + \
               loss_pro_1 + loss_pro_2

        return loss / batch_size

















