import os
import glob
import re
import numpy as np
from PIL import Image
import transforms3d.quaternions as txq
import tensorflow as tf

def data_load(data_dir, img_width, img_height, seq_num, mean_t, std_t, align_R, align_t, align_s):

#posenetはこの2行多分いらない
#pose_stats_filename = os.path.join(data_dir, 'pose_stats.txt') 
#mean_t, std_t = np.loadtxt(pose_stats_filename)

  imgs = []
  poses = np.empty([0, 12], dtype = np.float32)
  mean=np.array([5.009650708326967017e-01, 4.413125411911532625e-01, 4.458285283490354689e-01])
  std=np.array([4.329720281018845096e-02, 5.278270383679337097e-02, 4.760929057962018374e-02])
  std = np.sqrt(std)

  for seq in seq_num:
    imgs_seq = []
    seq = '{0:02}'.format(seq)
    # seq_dir = os.path.join("/content/gdrive/My Drive/dataset", "zemishitu_table", "sara_touki", "seq-%s" % seq, "")
    seq_dir = os.path.join("/home/komi/dataset", "7Scenes", "chess", "seq-%s" % seq, "")
    print(seq_dir)
    pose_filepath_list = glob.glob(seq_dir + '*.pose.txt')
    # pose = [np.loadtxt(pose_filepath).flatten()[:12] for pose_filepath in pose_filepath_list]

    for pose_filepath in pose_filepath_list:

      #pose
      pose = np.loadtxt(pose_filepath).flatten()[:12]
      poses = np.append(poses, np.array([pose]), axis=0)
      '''
      R_mat = np.loadtxt(pose_filepath)[:3,:3]
      q = txq.mat2quat(np.dot(align_R, R_mat))
      q *= np.sign(q[0])  # constrain to hemisphere
      q = qlog(q)
      
      t = np.loadtxt(pose_filepath)[:3,3:].flatten()
      t = t - align_t
  
      t = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()
      ###

      pose = np.hstack((t, q)) # pose[x, y, z, q1, q2, q3]
      poses = np.append(poses, np.array([pose]), axis=0)
      '''
      
      # image data
      # img_filepath = pose_filepath.replace('.pose.txt','.jpg')
      img_filepath = pose_filepath.replace('.pose.txt','.color.png')
      #print(pose_filepath)
      data = Image.open(img_filepath).resize(size=(img_width, img_height)) #size=(1024,768)
      data = data.resize((256, 256))
#      data = crop_center(data, 224, 224)
#      data = (data - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
      data = np.array(data, dtype=np.float32)
      data = data / 255.0
      imgs_seq.append(data)

  ### normalize img
#  imgs = tf.keras.applications.resnet.preprocess_input(imgs)

#  mean = mean * 255
#  imgs = ((imgs - mean.reshape(1, 1, 1, 3)))
#  imgs = imgs[..., ::-1]
    imgs_seq = ((imgs_seq - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3))
    #imgs.append(imgs_seq)
    imgs.extend(imgs_seq)
  imgs = np.array(imgs, dtype=np.float32)
  print('imgs_shape = {}'.format(imgs.shape))
  print('poses_shape1 = {}'.format(poses.shape))
  poses = process_poses(poses, mean_t, std_t, align_R, align_t, align_s)
  print('poses_shape2 = {}'.format(poses.shape))
  return imgs, poses

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def qlog(q):
  """
  Applies logarithm map to q
  :param q: (4,)
  :return: (3,)
  """
  if all(q[1:] == 0):
    q = np.zeros(3)
  else:
    q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
  return q

def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :param align_R: 3 x 3
  :param align_t: 3
  :param align_s: 1
  :return: processed poses (translation + quaternion) N x 7
  """
  poses_out = np.zeros((len(poses_in), 6))
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3]
    q = txq.mat2quat(np.dot(align_R, R))
    q *= np.sign(q[0])  # constrain to hemisphere
    q = qlog(q)
    poses_out[i, 3:] = q
    t = poses_out[i, :3] - align_t
    poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

  # normalize translation
  poses_out[:, :3] -= mean_t
  poses_out[:, :3] /= std_t
  return poses_out
