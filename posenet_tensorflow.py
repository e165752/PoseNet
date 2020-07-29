# -*- coding: utf-8 -*-
"""Posenet_tensorflow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aDtg-ij31_hxqZY3tufEnTHFcazq8xjl
"""

# !pip install -U tensorflow
# !pip install transforms3d

import tensorflow as tf
from tensorflow import keras
from data_loader import data_load, qlog 

# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

import math
import transforms3d.quaternions as txq
import numpy as np
import os
import sys
import os.path as osp
import pickle

data_dir = os.path.join('~', 'dataset', '7Scenes', 'chess')
img_width, img_height = 256, 256 #1024, 768 -> 320, 240 -> 160, 120
input_tensor = tf.keras.Input(shape=(img_height, img_width, 3))
# include_top=Falseとすることで，全結合層を除いたResNet50をインポートする
# weights=’imagenet’とすることで学習済みのResNet50が読み込める． weights=Noneだとランダムな初期値から始まる
ResNet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(img_width, img_height, 3))

#ResNet50.summary()

resnet_output_shape = ResNet50.output_shape[1:]
print(resnet_output_shape)

# resnet50の部分だけ誤差逆伝播しない，重み更新しない->False
for layer in ResNet50.layers:
  layer.trainable = False

x = ResNet50.output

x = AveragePooling2D((2, 2))(x)
x = Flatten()(x)
#x = Dropout(0.5)(x)
x = Dense(2048, activation='linear')(x) # linear関数なので
x = tf.keras.activations.relu(x) # 本家のコードでなぜかreluされている
x = Dropout(0.5)(x)
xyz = Dense(3, activation='linear')(x) # 同じくlinearに変更
wpqr = Dense(3, activation='linear')(x) # こっちもlinearに変更
pred = tf.concat([xyz, wpqr], 1)


model = Model(inputs=ResNet50.input, outputs=pred)
#model.summary()

# model save
json_string = model.to_json()
open(os.path.join('save_checkpoint','posenet_model.json'), 'w').write(json_string)

tensor = tf.random.uniform([3, 6])

tensor2 = tf.random.uniform([3, 6])

tf.slice(tensor, [0, 0], [-1, 3])

tf.slice(tensor, [0, 3], [-1, 3])

tf.cast(tensor, tf.float16)

mae = tf.keras.losses.MeanAbsoluteError()
print(mae(tf.slice(tensor, [0, 0], [-1, 3]), tf.slice(tensor, [0, 3], [-1, 3])))

def criterion_loss(sax, saq):
  def loss_function(targ, pred):
    # t_mae = mean_absolute_error(pred[:, :3], targ[:, :3])
    # q_mae = mean_absolute_error(pred[:, 3:], targ[:, 3:])

    # pred = tf.cast(pred, tf.float32)
    # targ = tf.cast(targ, tf.float32)
    t_mae = tf.keras.losses.MeanAbsoluteError() #tf.keras.losses.MAE
    targ_t = tf.slice(targ, [0, 0], [-1, 3])
    pred_t = tf.slice(pred, [0, 0], [-1, 3])
    t_mae = t_mae(targ_t, pred_t)

    r_mae = tf.keras.losses.MeanAbsoluteError()
    targ_r = tf.slice(targ, [0, 3], [-1, 3])
    pred_r = tf.slice(pred, [0, 3], [-1, 3])
    r_mae = r_mae(targ_r, pred_r)

    with tf.GradientTape(persistent=True) as tape:
      loss = tf.math.exp(-sax) * t_mae + sax + tf.math.exp(-saq) * r_mae + saq
    tape.gradient(loss, sax)
    tape.gradient(loss, saq)

    #tf.print('update: sax, saq = {}, {}'.format(sax.value(), saq.value()))

    #loss = tf.math.exp(-sax) * t_mae + sax + tf.math.exp(-saq) * r_mae + saq
    '''
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    loss_result = sess.run(loss)
    '''
    return loss
  return loss_function


#def mean_absolute_error(targ, pred):
#  AE = tf.reduce_sum(tf.abs(tf.subtract(targ, pred)), axis=1)
  # print(AE)
#  MAE = tf.divide(AE, targ.shape[1])
  # print(MAE)
#  return MAE

#init_sax = 0.0
#init_saq = -3.0

sax = tf.Variable(0.0, trainable=True, name='sax', dtype=tf.float32)
saq = tf.Variable(-3.0, trainable=True, name='saq', dtype=tf.float32) #hyperparameter: beta
tf.print('init: sax, saq = {}, {}'.format(sax.value(), saq.value()))
#tf.print(type(sax))
#print('sax_type: {}'.format(type(sax)))
#opt=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.0005)
#opt.minimize(criterion_loss, var_list=[sax,saq])

model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.0005),
              loss=criterion_loss(sax, saq),
              metrics=['accuracy'])

seq_num = [1, 2, 4, 6]
for seq in seq_num:
    print('{0:02}'.format(seq))

#def qlog(q):
#  """
#  Applies logarithm map to q
#  :param q: (4,)
#  :return: (3,)
#  """
#  if all(q[1:] == 0):
#    q = np.zeros(3)
#  else:
#    q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
#  return q

vo_stats = {}
#seq_num = [1, 2, 3, 4]
vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
std_t = np.ones(3)

train_img, train_pose = data_load(data_dir, img_width, img_height, seq_num, mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])

a = np.empty([0, 6])
b = [1, 2, 3, 4, 5, 6]
a = np.append(a, np.array([b]), axis=0)
a = np.append(a, np.array([b]), axis=0)

train_img = np.array(train_img).astype('float32')
train_pose = np.array(train_pose).astype('float32')

print(train_img.shape)

#pred_a = model.predict(train_img[-3:])
# pred_a = tf.convert_to_tensor(pred_a, dtype=tf.float32)
#pred_a = tf.constant(np.array(pred_a), dtype='float32')

# targ_a = tf.convert_to_tensor(targ_a, dtype=tf.float32)
#targ_a = tf.constant(np.array(train_pose[-3:]), dtype='float32')

#loss_function(pred_a, targ_a)

a = tf.random.uniform([3, 3])

b = tf.random.uniform([3, 3])

tf.concat([a, b], -1)

checkpoint_path = "save_checkpoint/cp_{epoch:03d}_L{loss:.04f}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True, save_best_only=True,
    period=10)

def error_of_metric_angle(targ_poses, pred_poses):
  
  t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
  q_criterion = quaternion_angular_error

  t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                       targ_poses[:, :3])])

  q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                       targ_poses[:, 3:])])
  return t_loss, q_loss

def quaternion_angular_error(q1, q2):
  """
  angular error between two quaternions
  :param q1: (4, )
  :param q2: (4, )
  :return:
  """
  d = abs(np.dot(q1, q2))
  d = min(1.0, max(-1.0, d))
  theta = 2 * np.arccos(d) * 180 / np.pi
  return theta

### call_back
class DisplayCallBack(tf.keras.callbacks.Callback):
  # コンストラクタ
  def __init__(self):
    self.last_acc, self.last_loss, self.last_val_acc, self.last_val_loss = None, None, None, None
    self.now_batch, self.now_epoch = None, None

    self.epochs, self.samples, self.batch_size = None, None, None
  
  def qexp(self, q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q

  def on_epoch_end(self, epoch, logs={}):
    output = model.predict(train_img[::20, :, :, :], verbose=0)
    target = train_pose[::20,:]

    #tf.print(train_img[::20, :, :, :].shape, train_pose[::20,:].shape)

    ####
    # normalize the predicted quaternions
    tf.print(output.shape)
    q = [self.qexp(p[3:]) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    q = [self.qexp(p[3:]) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))
    predict_result_filename = osp.join('predict_result', 'epoch_{0:03}.pkl'.format(epoch))
    with open(predict_result_filename, 'wb') as f:
        pickle.dump({'targ_poses': target, 'pred_poses': output}, f)
    print('{:s} written'.format(predict_result_filename))


    # un-normalize the predicted and target translations
    pose_m = np.zeros(3)  # optionally, use the ps dictionary to calc stats
    pose_s = np.ones(3)
    #pose_s = np.array([0.0000000, 0.0000000, 0.0000000])
    #pose_m = np.array([1.0000000, 1.0000000, 1.0000000])
    output[:, :3] = (output[:, :3] * pose_s) + pose_m
    target[:, :3] = (target[:, :3] * pose_s) + pose_m

    '''  
    batch_idx = 1  
    idx = [batch_idx]
    idx = idx[len(idx) / 2]

    # take the middle prediction
    output[idx, :] = output[len(output)/2]
    target[idx, :] = target[len(target)/2]
    '''
    ####

    t_loss, q_loss = error_of_metric_angle(target, output)

    tf.print('\nError in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
    'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree' \
    .format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))

    tf.print('sax, saq = {}, {}'.format(sax.value(), saq.value()))

  '''    
  def on_batch_end(self, batch, logs=None):
    tf.print('sax, saq = {}, {}'.format(sax.value(), saq.value()))
  '''
loss_callback = DisplayCallBack() 

#model.fit(train_img, train_pose, epochs=600, batch_size=256, validation_split=0.3, verbose=1, callbacks = [cp_callback, loss_callback])
model.fit(train_img, train_pose, epochs=700, batch_size=64, validation_split=0.3, verbose=1, callbacks = [cp_callback, loss_callback])

# sevenseansデータセットで学習させる(コードがうまくいってるかの検証，論文と比較)
# 写真撮影して試す(全部三次元再構成できるところ)
# accuracyの定義を自分でしないといけないかも
