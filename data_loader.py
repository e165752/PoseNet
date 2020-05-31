import os
import glob
import re
import numpy as np
from PIL import Image
import transforms3d.quaternions as txq

def data_load(data_dir, img_width, img_height, seq_num, mean_t, std_t, align_R, align_t, align_s):

#posenetはこの2行多分いらない
#pose_stats_filename = os.path.join(data_dir, 'pose_stats.txt') 
#mean_t, std_t = np.loadtxt(pose_stats_filename)

  for seq in seq_num:

    seq = '{0:02}'.format(seq)
    # seq_dir = os.path.join("/content/gdrive/My Drive/dataset", "zemishitu_table", "sara_touki", "seq-%s" % seq, "")
    seq_dir = os.path.join("/home/komi/dataset", "7Scenes", "chess", "seq-%s" % seq, "")
    print(seq_dir)
    pose_filepath_list = glob.glob(seq_dir + '*.pose.txt')
    # pose = [np.loadtxt(pose_filepath).flatten()[:12] for pose_filepath in pose_filepath_list]

    imgs = []
    poses = np.empty([0, 6], dtype = np.float32)

    for pose_filepath in pose_filepath_list:

      #pose
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

      ### normalize translation
      poses[:, :3] -= mean_t
      poses[:, :3] /= std_t
      ###

      # img data
      # img_filepath = pose_filepath.replace('.pose.txt','.jpg')
      img_filepath = pose_filepath.replace('.pose.txt','.color.png')
      #print(pose_filepath)
      data = Image.open(img_filepath).resize(size=(img_width, img_height)) #size=(1024,768)
      data = np.array(data, dtype=np.float32)
      # data = data / 255.0
      imgs.append(data)

  return imgs, poses


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
