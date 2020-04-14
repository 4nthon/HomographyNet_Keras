import glob
import os

import cv2
import keras
import numpy as np

#数据生成类
class DataGeneratorHomographyNet(keras.utils.Sequence):
    #功能：给Keras生成数据
    #构造函数 传入配置参数
    def __init__(self, img_paths, input_dim=(480, 480), batch_size=64, shuffle=True, mode='fit',
                 n_channels=2, label_dim=(8,), resize_dim=(800, 800)):
        #输入图片尺寸
        self.input_dim = input_dim
        #输入图片路径
        self.img_paths = img_paths
        #batchsize
        self.batch_size = batch_size
        #是否打乱顺序
        self.shuffle = shuffle
        #计算图片的索引
        self.indexes = np.arange(len(self.img_paths))
        #模式
        self.mode = mode
        #通道
        self.n_channels = n_channels
        #标签维度
        self.label_dim = label_dim
        #计算随机扰动范围
        self.rho = int(min(input_dim)) // 4
        #重设尺寸
        self.resize_dim = resize_dim
        
        self.on_epoch_end()

    def __len__(self):
        #功能：表示每个epoch的批数
        #batchsize除以图片数量下取整再转成整型
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        #功能：生成一个批次的数据
        # 生成批次的索引
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # 寻找列表的IDs
        img_paths_batch = [self.img_paths[k] for k in indexes]

        if self.mode == 'fit':
            data_batch, label_batch = self.__generate_batch(img_paths_batch)
            return data_batch, label_batch
        # elif self.mode == 'predict':
        #     return X

        else:
            raise AttributeError('模式参数应设置为fit或predict.')

    def on_epoch_end(self):
        #功能：在每个epoch之后更新索引
        self.indexes = np.arange(len(self.img_paths))
        #应用随机打乱顺序
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __generate_batch(self, img_paths_batch):
        #功能：生成包含batch_size个样本的数据
        # 初始化空的矩阵存放每个批次数据和标签
        data_batch = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        label_batch = np.empty((self.batch_size, *self.label_dim))

        # 生成数据
        for idx, img_path in enumerate(img_paths_batch):
            #图像标准化，重设尺寸
            img = self.__load_grayscale(img_path)
            #获取两种（扭曲的和没扭曲的）四边形的拼接图，单应性矩阵
            tmp_data, tmp_m = self.generate_img_stack(img)
            #这个图丢去训练
            data_batch[idx, ] = tmp_data
            #这个就是论文的核心概念：单应性矩阵就是要拟合的目标，这个操作是拉直
            label_batch[idx, ] = tmp_m.flatten()[:-1]
        #返回数据和标签
        return data_batch, label_batch

    def generate_img_stack(self, img):
        #获取图片的尺寸
        h, w = img.shape[0: 2]
        #生成四个点的随机扰动值 delta值
        shift = np.random.randint(-self.rho, self.rho, size=(4, 2))
        #生成加了扰动值的中心矩形框坐标
        crop_pnts = self.generate_4pnts((h, w))
        #秘技-双重扰动合并！
        new_pnts = np.add(crop_pnts, shift).astype(np.float32)
        #第一次扰动的坐标和第二次扰动以后的坐标对应求解单应性矩阵
        #也就是说第一次坐标就是矩形框坐标，可能原作者不是什么规规矩矩的人
        #所以要加一个微小的形变，第二次就是随机扰动让矩形框直接变形成其他四边形
        curr_m = cv2.getPerspectiveTransform(crop_pnts, new_pnts)
        #用单应性矩阵做透视变换
        warped = cv2.warpPerspective(img, curr_m, (w, h))
        #我猜变换完了可能是3通道的，取一个R通道
        warped = np.expand_dims(warped, axis=-1)
        #注意上面的变换是全图变换
        #这里要按照坐标把它裁剪出来
        #裁剪原图的
        crop_img_1 = img[int(min(crop_pnts[:, 1])): int(max(crop_pnts[:, 1])), int(min(crop_pnts[:, 0])): int(max(crop_pnts[:, 0])), :]
        #裁剪变换后的
        crop_img_2 = warped[int(min(crop_pnts[:, 1])): int(max(crop_pnts[:, 1])), int(min(crop_pnts[:, 0])): int(max(crop_pnts[:, 0])), :]

        # cv2.imshow("orig", img)
        # cv2.imshow("transformed", warped)
        # cv2.imshow("crop_1", crop_img_1)
        # cv2.imshow("crop_2", crop_img_2)
        # cv2.waitKey(0)
        #返回原图裁剪的四边形和变换后的四边形，注意这里被拼接在一起了，还有一个单应性矩阵被一块返回
        return np.concatenate((crop_img_1, crop_img_2), axis=-1), curr_m

    def generate_4pnts(self, orig_img_dim):
        #功能：生成加了扰动值的中心矩形框坐标
        #传入图像尺寸
        h, w = orig_img_dim
        #如果输入图片刚好是480,480，那么x、y应该正好是中心矩形的左上角坐标且加了一个随机扰动值
        x = np.random.randint(self.rho, w - self.rho - self.input_dim[1])
        y = np.random.randint(self.rho, h - self.rho - self.input_dim[0])
        #返回四个坐标 因为加了扰动，所以不可能是矩形或者平行四边形
        return np.asarray([[x, y],
                           [x + self.input_dim[1], y],
                           [x, y + self.input_dim[0]],
                           [x + self.input_dim[1], y + self.input_dim[0]]], dtype=np.float32)

    def __load_grayscale(self, img_path):
        #功能：获取灰度图，标准化，尺寸重设
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.resize_dim)
        img = img.astype(np.float32) / 255.
        #获取最后一个通道 R通道的
        img = np.expand_dims(img, axis=-1)

        return img

    @staticmethod
    def __load_rgb(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img
