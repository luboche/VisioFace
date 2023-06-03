"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    about V6:
    修改:仅当没有yolo框的时候，才可能利用tracker补框。
"""

from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from skimage import io

import glob
import time
import argparse
import sys
sys.path.append("/data1/jing_li/otherfiles/YOLOV8/Anti_UAV") 
from filterpy.kalman import KalmanFilter
from CMC import align_images,warp_pos

np.random.seed(0)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


# def iou_batch(bb_test, bb_gt):
#   """
#   From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#   """
#   bb_gt = np.expand_dims(bb_gt, 0)
#   bb_test = np.expand_dims(bb_test, 1) ####
  
#   xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
#   yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#   xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#   yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#   w = np.maximum(0., xx2 - xx1)
#   h = np.maximum(0., yy2 - yy1)
#   wh = w * h
#   o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
#     + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
#   return(o)  

def iou_batch(bb_test, bb_gt): # iou
    # 计算交集框的左上角和右下角坐标
    xx1 = np.maximum(bb_test[:, np.newaxis, 0], bb_gt[np.newaxis, :, 0])
    yy1 = np.maximum(bb_test[:, np.newaxis, 1], bb_gt[np.newaxis, :, 1])
    xx2 = np.minimum(bb_test[:, np.newaxis, 2], bb_gt[np.newaxis, :, 2])
    yy2 = np.minimum(bb_test[:, np.newaxis, 3], bb_gt[np.newaxis, :, 3])

    # 计算交集框的宽度和高度
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    # 计算交集框的面积
    intersection_area = w * h

    # 计算各自框的面积
    bb_test_area = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    bb_gt_area = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])

    # 计算并集面积
    union_area = bb_test_area[:, np.newaxis] + bb_gt_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou

def giou_batch(bb_test, bb_gt): # giou
    # 交集框
    ## 计算交集框的左上角和右下角坐标
    xx1 = np.maximum(bb_test[:, np.newaxis, 0], bb_gt[np.newaxis, :, 0])
    yy1 = np.maximum(bb_test[:, np.newaxis, 1], bb_gt[np.newaxis, :, 1])
    xx2 = np.minimum(bb_test[:, np.newaxis, 2], bb_gt[np.newaxis, :, 2])
    yy2 = np.minimum(bb_test[:, np.newaxis, 3], bb_gt[np.newaxis, :, 3])

    ## 计算交集框的宽度和高度
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    # 计算交集框的面积
    intersection_area = w * h

    # 计算各自框的面积
    bb_test_area = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    bb_gt_area = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])

    # 计算并集面积
    union_area_1 = bb_test_area[:, np.newaxis] + bb_gt_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area_1

    # 并集框
    ## 计算并集框的左上角和右下角坐标
    x1 = np.minimum(bb_test[:, np.newaxis, 0], bb_gt[np.newaxis, :, 0])
    y1 = np.minimum(bb_test[:, np.newaxis, 1], bb_gt[np.newaxis, :, 1])
    x2 = np.maximum(bb_test[:, np.newaxis, 2], bb_gt[np.newaxis, :, 2])
    y2 = np.maximum(bb_test[:, np.newaxis, 3], bb_gt[np.newaxis, :, 3])

    # 计算并集框的宽度和高度
    w_union = np.maximum(0, x2 - x1)
    h_union = np.maximum(0, y2 - y1)

    # 计算并集框的面积
    union_area_2 = w_union * h_union

    # 计算GIOU
    giou = iou - (union_area_2 - union_area_1) / union_area_2

    return iou


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area(no,just scale) and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0] # dx
  h = bbox[3] - bbox[1] # dy
  x = bbox[0] + w/2. # cx
  y = bbox[1] + h/2. # cy
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    # 协方差矩阵
    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.

    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0 # 连续hit次数（加乘减除）
    self.cumulative_hit_streak = 0 # 累计命中次数
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.cumulative_hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self,affine_matrix):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    A: affine_matrix
    """
    if((self.kf.x[6]+self.kf.x[2])<=0): # 面积缩减为0
      self.kf.x[6] *= 0.0
    self.kf.predict() # 预测，得到仿射变换之前的state
    # 仿射变换
    bbox = convert_x_to_bbox(self.kf.x[:4])
    bbox = warp_pos(bbox.reshape(1,-1), affine_matrix)
    self.kf.x[:4] = convert_bbox_to_z(bbox)

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = int(self.hit_streak / 1.4)
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,matrix_func=iou_batch,iou_threshold = 0.05):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = matrix_func(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age # 最大失联帧数
    self.min_hits = min_hits # 最小匹配帧数
    self.iou_threshold = iou_threshold
    self.trackers = [] # 跟踪器
    self.frame_count = 0 # 帧数计数器
    self.last_image = [] # 上一帧

  def update(self, image=[],dets=np.empty((0, 5)),cmc=True,bad_frame=[],metrix='giou'):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    if metrix == 'iou':
      metrix_batch = iou_batch
    elif metrix == 'giou':
      metrix_batch = giou_batch
  
    if self.frame_count in bad_frame:
      self.frame_count
    if cmc:
      # CMC:camera motion compensation
      _, warp_matrix =  align_images(self.last_image, image,mask_or_crop='crop',ECC_or_feature='ECC')
      # try :
      #   _, warp_matrix =  align_images(self.last_image, image,mask_or_crop='crop',ECC_or_feature='feature')
      # except:
      #   _, warp_matrix =  align_images(self.last_image, image,mask_or_crop='crop',ECC_or_feature='ECC')
    else:
      warp_matrix = np.eye(2, 3, dtype=np.float32)
    self.last_image = image 

    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5)) # 存储预测位置，多个trackers
    to_del = [] # 存储预测位置包含NaN的跟踪器索引
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict(warp_matrix)[0].reshape(-1)
      # if cmc:
      #   pos = warp_pos(pos.reshape(1,-1), warp_matrix) # CMC  

      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # 过滤NaN行
    for t in reversed(to_del):
      self.trackers.pop(t) # 移除跟踪器
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,metrix_batch,self.iou_threshold) # 匈牙利算法目标关联
    '''matched:list of tuple(m[0], m[1]),m[0]是dets index,m[0]是trackers index
    '''
    # update matched trackers with assigned detections
    for m in matched: # [d,t]
      # 如果是重匹配的tracker
      if self.trackers[m[1]].time_since_update > 0:
        self.trackers[m[1]].hit_streak = self.trackers[m[1]].cumulative_hit_streak
      self.trackers[m[1]].update(dets[m[0], :])


    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i,:])
      self.trackers.append(trk)
      # update matched
      matched = np.append(matched, np.array([[i,len(self.trackers)-1]]), axis=0)
    
    

    # get predict results
    # ## get predict results from detector
    # for d in range(dets.shape[0]):
    #   ret.append(np.concatenate((dets[int(d),:4].reshape(-1),[trk.id+1,int(d), ])).reshape(1,-1)) # 增加返回score
    ## get predict results from tracker    
    pop_list = []
    for t, trk in enumerate(self.trackers):
      # tracker in matches
      if t in matched[:,1]:
        d_index = matched[np.where(matched[:, 1] == t)[0],0] # detection index
        ## d
        try:
          ret.append(np.concatenate((dets[d_index,:4].reshape(-1),[trk.id+1,d_index,0])).reshape(1,-1)) # 增加返回score
        except:
          d_index
        ## t
        # x_t = trk.get_state()[0]
        # iou = metrix_batch(dets[d_index,:4].reshape(-1,4) , x_t.reshape(-1,4))[0,0]
        # ret.append(np.concatenate((x_t,[trk.id+1,-1,iou])).reshape(1,-1)) # 增加返回iou
      # tracker not in matches
      # elif (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits) and np.size(dets)==0: # 有效跟踪器
      elif (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # 有效跟踪器
        x_t = trk.get_state()[0]
        ret.append(np.concatenate((x_t,[trk.id+1,-1,0])).reshape(1,-1)) # 增加返回iou
      # append pop_list
      if(trk.time_since_update > self.max_age):
        pop_list.append(t)
        
    # remove dead tracklet(reverse first)
    for t in pop_list[::-1]:
      self.trackers.pop(t)
    if(len(ret)>0):
      return np.concatenate(ret)
    '''ret:[x1,y1,x2,y2,t_id,d_id],d_id==-1表示由tracker补全'''
    return np.empty((0,5))


    # i = len(self.trackers)
    # for trk in reversed(self.trackers):
    #   # update / new -> get_state:当前的状态,更新过
    #   d = trk.get_state()[0]
    #   if (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits): # 有效跟踪器 
    #     ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
    #   # elif trk.age < 1: # 认为可以使用新的跟踪器
    #   #   ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
    #   i -= 1
    #   # remove dead tracklet
    #   if(trk.time_since_update > self.max_age):
    #     self.trackers.pop(i)
    # if(len(ret)>0):
    #   return np.concatenate(ret)
    # return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    ####################################
    #create instance of the SORT tracker
    ####################################
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) 
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] # 提取序列名称
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          ax1.axis('off')
          # plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          # 添加代码：保存文件至'mot_benchmark_result/'文件夹下
          output_dir = os.path.join('mot_benchmark_result', phase, seq, 'img1') # img1是子文件夹名称
          os.makedirs(output_dir, exist_ok=True)
          output_path = os.path.join(output_dir, '%06d.png' % frame)
          plt.savefig(output_path,bbox_inches='tight',pad_inches=0)
          # 清空轴
          ax1.cla()

  # print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
