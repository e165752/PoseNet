from data_loader import data_load, qlog
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import model_from_json

import numpy as np
import os
import glob
import sys

args = sys.argv
ckpt_filename = args[1]

vo_stats = {}

data_dir = os.path.join('~', 'dataset', '7Scenes', 'chess')
seq_num = [1, 2]

for seq in seq_num:
    print('{0:02}'.format(seq))

vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
std_t = np.ones(3)

def loss_function(targ_poses, pred_poses):
  t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
  q_criterion = quaternion_angular_error 

  t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                       targ_poses[:, :3])])

  q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                       targ_poses[:, 3:])])  
#  pred_t = tf.slice(pred_poses, [0, 0], [-1, 3])
#  targ_t = tf.slice(targ_poses, [0, 0], [-1, 3])
#  pred_q = tf.slice(pred_poses, [0, 3], [-1, 3])
#  targ_q = tf.slice(targ_poses, [0, 3], [-1, 3])

#  t_criterion(pred_t, targ_t)
#  q_criterion(pred_q, targ_q)

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

###########
model_filename = 'posenet_model.json'
json_string = open(os.path.join('save_checkpoint', model_filename)).read()
model = model_from_json(json_string)
checkpoint_path = '/home/komi/master_thesis/PoseNet/save_checkpoint/{}'.format(ckpt_filename)
model.load_weights(checkpoint_path)

model_inputshape = model.input_shape
img_width, img_height = model_inputshape[2], model_inputshape[1]
#print(img_width, img_height)

test_img, test_pose = data_load(data_dir, img_width, img_height, seq_num, mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])

test_img = np.array(test_img).astype('float32')
test_pose = np.array(test_pose).astype('float32')

pred_poses = model.predict(test_img, verbose=2)
t_loss, q_loss = loss_function(test_pose, pred_poses)

print(pred_poses.shape)

print(t_loss.shape, q_loss.shape)

print('Error in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
    'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree' \
    .format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))
