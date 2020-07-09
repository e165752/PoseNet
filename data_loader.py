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

  imgs = []
  poses = np.empty([0, 6], dtype = np.float32)
  for seq in seq_num:

    seq = '{0:02}'.format(seq)
    # seq_dir = os.path.join("/content/gdrive/My Drive/dataset", "zemishitu_table", "sara_touki", "seq-%s" % seq, "")
    seq_dir = os.path.join("/home/komi/dataset", "7Scenes", "chess", "seq-%s" % seq, "")
    print(seq_dir)
    pose_filepath_list = glob.glob(seq_dir + '*.pose.txt')
    # pose = [np.loadtxt(pose_filepath).flatten()[:12] for pose_filepath in pose_filepath_list]

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

      # image data
      # img_filepath = pose_filepath.replace('.pose.txt','.jpg')
      img_filepath = pose_filepath.replace('.pose.txt','.color.png')
      #print(pose_filepath)
      data = Image.open(img_filepath).resize(size=(img_width, img_height)) #size=(1024,768)
      data = data.resize((256, 256))
      data = crop_center(data, 224, 224)
#      data = (data - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
      data = np.array(data, dtype=np.float32)
      # data = data / 255.0
      imgs.append(data)

  ### normalize img
#  mean=np.array([0.485, 0.456, 0.406])
#  std=np.array([0.229, 0.224, 0.225])
  mean=np.array([5.009650708326967017e-01, 4.413125411911532625e-01, 4.458285283490354689e-01])
  std=np.array([4.329720281018845096e-02, 5.278270383679337097e-02, 4.760929057962018374e-02])
  std = np.sqrt(std)
  imgs = ((imgs - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3))
  ### normalize translation
  poses[:, :3] -= mean_t
  poses[:, :3] /= std_t

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
