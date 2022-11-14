
import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow_addons.layers as tfa_layers

import torch        #@HRn
import torch.nn as nn       #@HRn


import ops as hdrnet_ops


class HalfInstanceNorm(nn.Module):
  """ 
      ....

  """
  def __init__(self):
    super().__init__()
    # self.size = input
    # self.InsNorm = nn.InstanceNorm2d()

  def forward(self , input):
    input_1 , input_2 = input[: , : int(input.size(1)/2) , : , :] , input[: , int(input.size(1)/2) : , : , :]
    input_1_InsNorm = nn.InstanceNorm2d(input_1.size(1))(input_1) 
    output = torch.cat((input_1_InsNorm , input_2) , dim = 1)
    return output


# class HalfInstance(tf.keras.Model):
#   '''Half Instance Normalization
#   '''
#   def __init__(self):
#     super().__init__()
#     self.in1 = tfa_layers.InstanceNormalization(axis=3)

#   def call(self, inputs):
#     channel = tf.shape(inputs)[3]
#     inputs_1, inputs_2 = tf.split(inputs, num_or_size_splits=2, axis=3)
#     inputs_2 = self.in1(inputs_2)
#     x = tf.concat([inputs_1, inputs_2], -1)
#     return x

class Identity(nn.Module):    #@HRn
  """
      ...

  """
  def __init__(self):
    super().__init__()

  def forward(self , input):
    output = nn.Identity()(input)
    return output


# class Identity(tf.keras.Model):
#   '''identity
#   '''
#   def __init__(self):
#     super().__init__()

#   def call(self, inputs):
#     return tf.identity(inputs)

def get_norm(norm_type):    #@HRn
  if norm_type == 'bn':
    return nn.BatchNorm2d
  elif norm_type == 'syncbn':
    return nn.SyncBatchNorm
  elif norm_type == 'hi':
    return HalfInstanceNorm
  elif norm_type == 'in':
    return nn.InstanceNorm2d
  elif norm_type == 'none':
    return Identity
  else:
    raise ValueError(f'Unrecognized norm_type {norm_type}')


def make_ShuffleM1PReLUPyramid3RGrid(channel=64, norm_type="syncbn", depth=6, gz=1, low_res='down', stage=3, group=1, output_channel=3):
  """We update model_1 and model_2 first and then update three model together, see supplementary for detailed explaination"""
  model_1 = ModelOne(channel=channel, norm_type=norm_type, depth=depth)
  model_2 = ModelTwo(channel=channel, norm_type=norm_type, gz=gz, low_res=low_res, stage=stage, group=channel)
  model_3 = ModelThree(channel=channel, norm_type=norm_type, output_channel=output_channel)
  return [model_1, model_2, model_3]


class ResConvBlock(tf.keras.Model):
  def __init__(self, channel=64, norm_type="syncbn"):
    super().__init__()
    NormLayer = get_norm(norm_type)
    self.convblock = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          channel*5, 1, padding='same',
          kernel_initializer='he_normal', trainable=True, use_bias=True),
      NormLayer(),
      tf.keras.layers.PReLU(shared_axes=[1, 2]),
      tf.keras.layers.DepthwiseConv2D(
          3, padding='same',
          kernel_initializer='he_normal', trainable=True, use_bias=True),
      NormLayer(),
      tf.keras.layers.PReLU(shared_axes=[1, 2]),
      tf.keras.layers.Conv2D(
          channel, 1, padding='same',
          kernel_initializer='he_normal', trainable=True, use_bias=True),
      NormLayer(),
      tf.keras.layers.PReLU(shared_axes=[1, 2]),
      ])

  def call(self, inputs, training=True):
    x = inputs
    output = x + self.convblock(x, training=training)
    return output

def space_to_depth(input):   #@HRn

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


class ShufflePyramidDecom(tf.keras.Model):
  def __init__(self, num_high=3):
    super().__init__()

    self.interpolate_mode = 'bicubic'
    self.num_high = num_high
    # self.downsample = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
    #                                padding='same')
    # self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

  def call(self, img, training=True):
    current = img
    pyr = []
    pyr.append(current)
    for i in range(self.num_high):
      down = tf.nn.space_to_depth(current, 2)
      pyr.append(down)
      current = down
    # pyr.append(current)
    return pyr


class Model_one(nn.Module):
  def __init__(self , channel = 64 , norm_type = "syncbn " , depth = 3):
    super().__init()
    self.channel = channel
    self.shuffle_decom = ShufflePyramidDecom(3)



class ModelOne(tf.keras.Model):
  def __init__(self, channel=64, norm_type="syncbn", depth=3):
    super().__init__()
    self.channel=channel

    self.shuffle_decom = ShufflePyramidDecom(3)
    self.conv_pyramid = [tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                      channel, 3, padding='same',
                      kernel_initializer='he_normal',
                      trainable=True,
                      use_bias=True),
                    ResConvBlock(channel),
                    ])
        for i in range(2)]
    self.conv_pyramid.append(tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                      channel*2, 3, padding='same',
                      kernel_initializer='he_normal',
                      trainable=True,
                      use_bias=True),
                    ResConvBlock(channel*2),
                      ]))
    self.conv_pyramid.append(tf.keras.Sequential([
                      tf.keras.layers.Conv2D(
                            channel*4, 3, padding='same',
                            kernel_initializer='he_normal',
                            trainable=True,
                            use_bias=True),
                      ResConvBlock(channel*4),
                      ]))

  def call(self, inputs, training=True):
    pyr = self.shuffle_decom(inputs)
    output = []
    for i, image in enumerate(pyr):
      output.append(self.conv_pyramid[i](image, training=training))
    return output, pyr


class ModelTwo(tf.keras.Model):
  def __init__(self, channel=64, norm_type='syncbn', gz=1, low_res='down', stage=3, group=64):
    super().__init__()
    self.channel = channel
    self.group = group
    self.stage = stage
    self.gz = gz
    self.n_in = 1
    self.n_out = 2

    NormLayer = get_norm(norm_type)
    if low_res == 'down':
      self.low_res_blocks = [Policy(channel=channel, n_in=self.n_in,
                                       n_out=self.n_out, gz=self.gz*self.group,
                                        stage=self.stage) for i in range(4)]
    elif low_res == 'downavg2':
      self.low_res_blocks = [tf.keras.Sequential([
                         tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding='same'),
                          Policy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group,
                                        stage=self.stage)
                            ]) for i in range(2)]
      self.low_res_blocks.append(Policy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group*2,
                                        stage=self.stage))
      self.low_res_blocks.append(Policy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group*4,
                                        stage=self.stage))
    elif low_res == 'ldownavg2':
      self.low_res_blocks = [tf.keras.Sequential([
                         tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding='same'),
                          LargePolicy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group,
                                        stage=self.stage)
                            ]) for i in range(2)]
      self.low_res_blocks.append(LargePolicy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group*2,
                                        stage=self.stage))
      self.low_res_blocks.append(LargePolicy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group*4,
                                        stage=self.stage))
    elif low_res == 'downavg4':
      self.low_res_blocks = [tf.keras.Sequential([
                         tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4),
                                         padding='same'),
                          Policy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group,
                                        stage=self.stage)
                            ]) for i in range(2)]
      self.low_res_blocks.append(Policy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group*2,
                                        stage=self.stage))
      self.low_res_blocks.append(Policy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group*4,
                                        stage=self.stage))
    elif low_res == 'ldownavg4':
      self.low_res_blocks = [tf.keras.Sequential([
                         tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4),
                                         padding='same'),
                          LargePolicy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group,
                                        stage=self.stage)
                            ]) for i in range(2)]
      self.low_res_blocks.append(LargePolicy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group*2,
                                        stage=self.stage))
      self.low_res_blocks.append(LargePolicy(channel=channel, n_in=self.n_in,
                                        n_out=self.n_out, gz=self.gz*self.group*4,
                                        stage=self.stage))
    elif low_res == 'unet':
      self.low_res_blocks = [tf.keras.Sequential([
                make_unet_upBilinear(channel=channel, n_in=self.n_in,
                                             n_out=self.n_out, gz=self.gz*self.group),
                ]) for i in range(4)]
    self.bilateral_blocks = [tf.keras.Sequential([
          NormLayer(),
          tf.keras.layers.PReLU(shared_axes=[1, 2]),
          # tf.keras.layers.Activation('relu'),
        ]) for i in range(4)]
    self.gate = [tf.Variable(0., trainable=True) for i in range(4)]

    self.conv4_1 = tf.keras.Sequential([
        ResConvBlock(channel*4),
        ResConvBlock(channel*4),
                      ])
    self.conv4_2 = tf.keras.Sequential([
        ResConvBlock(channel*4),
        ResConvBlock(channel*4),
                      ])
    self.conv3_1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
                            channel*2, 3, padding='same',
                            kernel_initializer='he_normal',
                            trainable=True),
        NormLayer(),
        ResConvBlock(channel*2),
        ResConvBlock(channel*2),
                      ])
    self.conv3_2 = tf.keras.Sequential([
        ResConvBlock(channel*2),
        ResConvBlock(channel*2),
                      ])
    self.conv2_1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
                            channel, 3, padding='same',
                            kernel_initializer='he_normal',
                            trainable=True),
        NormLayer(),
        ResConvBlock(channel),
        ResConvBlock(channel),
                      ])
    self.conv2_2 = tf.keras.Sequential([
        ResConvBlock(channel),
        ResConvBlock(channel),
                      ])
    self.conv1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
                            channel, 1, padding='same',
                            kernel_initializer='he_normal',
                            trainable=True),
        NormLayer(),
    ])
    self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')


  def call(self, inputs, training=True):
    inputs, pyr = inputs
    output = []
    image_1, image_2, image_3, image_4 = inputs
    # level_4
    image_4 = self.conv4_1(image_4)
    group = self.group * 4
    content_feature = image_4
    grid_coefficients = self.low_res_blocks[-1](content_feature, training=training)
    grid_coefficients = tf.stack(
        tf.split(grid_coefficients, group, axis=3), axis=5)
    post_image = []
    for j in range(group):
      post_image_j = hdrnet_ops.bilateral_slice_apply(
              grid_coefficients[:, :, :, :, :, j],
              tf.clip_by_value(content_feature[:, :, :, j], 0, 1),
              content_feature[:, :, :, j:j+1],
              has_offset=True)
      post_image.append(post_image_j)
    post_image = tf.concat(post_image, axis=3)
    post_image = self.bilateral_blocks[3](post_image, training=training)
    post_image = tf.clip_by_value(post_image, -1, 1)
    image_4 = image_4 + post_image*self.gate[3]
    image_4 = self.conv4_2(image_4)

    # level_3
    image_3 = self.conv3_1(tf.concat([image_3, self.upsample(image_4)], 3))
    group = self.group * 2
    content_feature = image_3
    grid_coefficients = self.low_res_blocks[-2](content_feature, training=training)
    grid_coefficients = tf.stack(
        tf.split(grid_coefficients, group, axis=3), axis=5)
    post_image = []
    for j in range(group):
      post_image_j = hdrnet_ops.bilateral_slice_apply(
              grid_coefficients[:, :, :, :, :, j],
              tf.clip_by_value(content_feature[:, :, :, j], 0, 1),
              content_feature[:, :, :, j:j+1],
              has_offset=True)
      post_image.append(post_image_j)
    post_image = tf.concat(post_image, axis=3)
    post_image = self.bilateral_blocks[2](post_image, training=training)
    post_image = tf.clip_by_value(post_image, -1, 1)
    image_3 = image_3 + post_image*self.gate[2]
    image_3 = self.conv3_2(image_3)

    # level_2
    image_2 = self.conv2_1(tf.concat([image_2, self.upsample(image_3)], 3))
    group = self.group
    content_feature = image_2
    grid_coefficients = self.low_res_blocks[-3](content_feature, training=training)
    grid_coefficients = tf.stack(
        tf.split(grid_coefficients, group, axis=3), axis=5)
    # grid_coefficients, grid_coefficients_1 = tf.split(grid_coefficients, 2, axis=-1)
    post_image = []
    for j in range(group):
      post_image_j = hdrnet_ops.bilateral_slice_apply(
              grid_coefficients[:, :, :, :, :, j],
              tf.clip_by_value(content_feature[:, :, :, j], 0, 1),
              content_feature[:, :, :, j:j+1],
              has_offset=True)
      post_image.append(post_image_j)
    post_image = tf.concat(post_image, axis=3)
    post_image = self.bilateral_blocks[1](post_image, training=training)
    post_image = tf.clip_by_value(post_image, -1, 1)
    image_2 = image_2 + post_image*self.gate[1]
    image_2 = self.conv2_2(image_2)

    # level_1
    image_1 = self.conv1(tf.concat([image_1, self.upsample(image_2)], 3))
    group = self.group
    content_feature = image_1
    grid_coefficients = self.low_res_blocks[0](content_feature, training=training)
    grid_coefficients = tf.stack(
        tf.split(grid_coefficients, group, axis=3), axis=5)
    post_image = []
    for j in range(group):
      post_image_j = hdrnet_ops.bilateral_slice_apply(
              grid_coefficients[:, :, :, :, :, j],
              tf.clip_by_value(content_feature[:, :, :, j], 0, 1),
              content_feature[:, :, :, j:j+1],
              has_offset=True)
      post_image.append(post_image_j)
    post_image = tf.concat(post_image, axis=3)
    post_image = self.bilateral_blocks[0](post_image, training=training)
    post_image = tf.clip_by_value(post_image, -1, 1)
    image_1 = image_1 + post_image*self.gate[0]

    output.append(image_1)
    output.append(image_2)
    output.append(image_3)
    output.append(image_4)

    return output, pyr


class ModelThree(tf.keras.Model):
  def __init__(self, channel=64, norm_type="syncbn", depth=3, output_channel=3):
    super().__init__()
    self.channel=channel

    self.conv = tf.keras.Sequential([
                      ResConvBlock(channel),
                      tf.keras.layers.Conv2D(
                            output_channel, 3, padding='same',
                            kernel_initializer='he_normal',
                            trainable=True,
                            use_bias=True),
                      ])

  def call(self, inputs, training=True):
    inputs, pyr = inputs
    output = []
    final_output = self.conv(inputs[0])
    return pyr[0] - final_output


class Policy(tf.keras.Model):
  def __init__(self, channel=64, norm_type="syncbn", n_in=65, n_out=64, gz=1, stage=3):
    super().__init__()
    self.n_in = n_in
    self.n_out = n_out
    self.gz = gz
    self.stage = stage
    NormLayer = get_norm(norm_type)
    self.low_dense1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            channel, 3, padding='same',
            kernel_initializer='he_normal',
            trainable=True, use_bias=True),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
    ])

    self.low_blocks_1 = tf.keras.Sequential([
        ResConvBlock(channel=channel, norm_type=norm_type)
        for i in range(3)
    ])
    if self.stage > 1:
      self.low_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
      self.low_blocks_2 = tf.keras.Sequential([
          ResConvBlock(channel=channel, norm_type=norm_type)
          for i in range(3)
      ])
    if self.stage > 2:
      self.low_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
      self.low_blocks_3 = tf.keras.Sequential([
          ResConvBlock(channel=channel, norm_type=norm_type)
          for i in range(3)
      ])
    if self.stage > 3:
      self.low_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
      self.low_blocks_4 = tf.keras.Sequential([
          ResConvBlock(channel=channel, norm_type=norm_type)
          for i in range(3)
      ])
    self.low_dense2 = tf.keras.layers.Conv2D(
      self.gz*self.n_in*self.n_out, 1, padding='same', kernel_initializer='he_normal',trainable=True,
        use_bias=True)

  def call(self, inputs, training=True):
    x = self.low_dense1(inputs, training=training)
    # tf.print(x.shape)
    x = self.low_blocks_1(x, training=training)
    if self.stage > 1:
      x = self.low_pool_1(x)
      x = self.low_blocks_2(x, training=training)
      # tf.print(x.shape)
    if self.stage > 2:
      x = self.low_pool_2(x)
      x = self.low_blocks_3(x, training=training)
    if self.stage > 3:
      x = self.low_pool_3(x)
      x = self.low_blocks_4(x, training=training)
      # tf.print(x.shape)
    output = self.low_dense2(x, training=training)
    output = tf.stack(
        tf.split(output, self.n_out * self.n_in, axis=3), axis=4)
    return output


class LargePolicy(tf.keras.Model):
  def __init__(self, channel=64, norm_type="syncbn", n_in=65, n_out=64, gz=1, stage=3):
    super().__init__()
    self.n_in = n_in
    self.n_out = n_out
    self.gz = gz
    self.stage = stage
    NormLayer = get_norm(norm_type)
    self.low_dense1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            channel, 3, padding='same',
            kernel_initializer='he_normal',
            trainable=True, use_bias=True),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
    ])

    self.low_blocks_1 = tf.keras.Sequential([
        ResConvBlock(channel=channel, norm_type=norm_type)
        for i in range(3)
    ])
    if self.stage > 1:
      self.low_pool_1 = tf.keras.Sequential([
          tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
          tf.keras.layers.Conv2D(
              channel*2, 3, padding='same',
              kernel_initializer='he_normal',
              trainable=True, use_bias=True),
          NormLayer(),
          tf.keras.layers.PReLU(shared_axes=[1, 2]),
      ])
      self.low_blocks_2 = tf.keras.Sequential([
          ResConvBlock(channel=channel*2, norm_type=norm_type)
          for i in range(3)
      ])
    if self.stage > 2:
      self.low_pool_2 = tf.keras.Sequential([
          tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
          tf.keras.layers.Conv2D(
              channel*4, 3, padding='same',
              kernel_initializer='he_normal',
              trainable=True, use_bias=True),
          NormLayer(),
          tf.keras.layers.PReLU(shared_axes=[1, 2]),
      ])
      self.low_blocks_3 = tf.keras.Sequential([
          ResConvBlock(channel=channel*4, norm_type=norm_type)
          for i in range(3)
      ])
    if self.stage > 3:
      self.low_pool_3 = tf.keras.Sequential([
          tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
          tf.keras.layers.Conv2D(
              channel*8, 3, padding='same',
              kernel_initializer='he_normal',
              trainable=True, use_bias=True),
          NormLayer(),
          tf.keras.layers.PReLU(shared_axes=[1, 2]),
      ])
      self.low_blocks_4 = tf.keras.Sequential([
          ResConvBlock(channel=channel*8, norm_type=norm_type)
          for i in range(3)
      ])
    self.low_dense2 = tf.keras.layers.Conv2D(
      self.gz*self.n_in*self.n_out, 1, padding='same', kernel_initializer='he_normal',trainable=True,
        use_bias=True)

  def call(self, inputs, training=True):
    x = self.low_dense1(inputs, training=training)
    # tf.print(x.shape)
    x = self.low_blocks_1(x, training=training)
    if self.stage > 1:
      x = self.low_pool_1(x)
      x = self.low_blocks_2(x, training=training)
      # tf.print(x.shape)
    if self.stage > 2:
      x = self.low_pool_2(x)
      x = self.low_blocks_3(x, training=training)
    if self.stage > 3:
      x = self.low_pool_3(x)
      x = self.low_blocks_4(x, training=training)
      # tf.print(x.shape)
    output = self.low_dense2(x, training=training)
    output = tf.stack(
        tf.split(output, self.n_out * self.n_in, axis=3), axis=4)
    return output





class StaticPolicyNetwork(tf.keras.Model):
  def __init__(self, input_channel=64, channel=64, norm_type='syncbn', n_in=65, n_out=64, gz=1, stage=3, group=64):
    # super().__init__()
    self.n_in = n_in
    self.n_out = n_out
    self.gz = gz
    self.stage = stage
    self.group=group
    NormLayer = get_norm(norm_type)

    inputs = tf.keras.layers.Input(shape=(None, None, input_channel),
                                 name='inputs')
    x = tf.keras.layers.Conv2D(
          channel, 3, padding='same',
          kernel_initializer='he_normal',
          trainable=True, use_bias=True)(inputs)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    for i in range(3):
        x = StaticResConvBlock(channel=channel, norm_type=norm_type)(x)

    if self.stage > 1:
      x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
      x = tf.keras.layers.Conv2D(
          channel*2, 3, padding='same',
          kernel_initializer='he_normal',
          trainable=True, use_bias=True)(x)
      for i in range(3):
        x = StaticResConvBlock(channel=channel*2, norm_type=norm_type)(x)

    if self.stage > 2:
      x = tf.keras.layers.Conv2D(
          channel*4, 3, padding='same',
          kernel_initializer='he_normal',
          trainable=True, use_bias=True)(x)
      x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
      for i in range(3):
        x = StaticResConvBlock(channel=channel*4, norm_type=norm_type)(x)

    x = tf.keras.layers.Conv2D(
      self.gz*self.n_in*self.n_out*self.group, 1, padding='same', kernel_initializer='he_normal',trainable=True,
        use_bias=True)(x)
    # tf.print(x.shape)
    # x = tf.stack(
    #     tf.split(x, self.n_out * self.n_in * self.group, axis=3), axis=4)
    b, h, w, _ = tf.shape(x)
    x = tf.reshape(x, [b, h, w, self.group, self.n_in*self.n_out])
    super(StaticPolicyNetwork, self).__init__(
        inputs=inputs, outputs=x)


class StaticShufflePyramidDecom(tf.keras.Model):
  def __init__(self, num_high=3):
    inputs = tf.keras.layers.Input(shape=(None, None, 3),
                                 name='inputs')
    self.interpolate_mode = 'bicubic'
    self.num_high = num_high

    current = inputs
    pyr = []
    pyr.append(current)
    for i in range(self.num_high):
      down = tf.nn.space_to_depth(current, 2)
      pyr.append(down)
      current = down
    super(StaticShufflePyramidDecom, self).__init__(
        inputs=inputs, outputs=pyr)


class StaticResConvBlock(tf.keras.Model):
  def __init__(self, channel=64, norm_type="syncbn"):
    # super().__init__()
    NormLayer = get_norm(norm_type)

    inputs = tf.keras.layers.Input(shape=(None, None, channel), name='inputs')
    conv1 = tf.keras.Sequential([
          tf.keras.layers.Conv2D(
                channel*5, 1, padding='same',
                kernel_initializer='he_normal', trainable=True, use_bias=True),
          NormLayer(),
          tf.keras.layers.PReLU(shared_axes=[1, 2]),
          tf.keras.layers.DepthwiseConv2D(
                3, padding='same',
                kernel_initializer='he_normal', trainable=True, use_bias=True),
          NormLayer(),
          tf.keras.layers.PReLU(shared_axes=[1, 2]),
          tf.keras.layers.Conv2D(
                channel, 1, padding='same',
                kernel_initializer='he_normal', trainable=True, use_bias=True),
          NormLayer(),
          tf.keras.layers.PReLU(shared_axes=[1, 2]),
        ])(inputs)
    x = conv1 + inputs

    super(StaticResConvBlock, self).__init__(
        inputs=inputs, outputs=x)



