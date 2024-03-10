#!/opt/homebrew/bin/python3

import numpy as np
import tensorflow as tf
import cv2
import open3d as o3d
import pywavefront
#import tflite_runtime as tflite

k_shp_w_idx = 1
k_shp_h_idx = 0
k_dtct_w_idx = 0
k_dtct_h_idx = 1

class Anchor:
    def __init__(self, input_map_shape, feature_map_size, x, y):
        self.input_map_size = input_map_shape
        self.feature_map_size = feature_map_size
        self.x_scale = input_map_shape[k_shp_w_idx] / feature_map_size
        self.y_scale = input_map_shape[k_shp_h_idx] / feature_map_size
        self.center_x = (x + 0.5) * self.x_scale
        self.center_y = (y + 0.5) * self.y_scale

class Bbox:
    def __init__(self, dtype):
        self.tplft = np.zeros(2, dtype = dtype)
        self.btrht = np.zeros(2, dtype = dtype)
    def get_center(self):
        _x = (self.btrht[k_dtct_w_idx] + self.tplft[k_dtct_w_idx]) / 2.0
        _y = (self.btrht[k_dtct_h_idx] + self.tplft[k_dtct_h_idx]) / 2.0
        return (_x, _y)
    def get_width(self):
        return self.btrht[k_dtct_w_idx] - self.tplft[k_dtct_w_idx]
    def get_height(self):
        return self.btrht[k_dtct_h_idx] - self.tplft[k_dtct_h_idx]

class Frustum:
    def __init__(self, hfov, d_near, d_far, aspect_ratio):
        self.d_near = d_near
        self.d_far = d_far
        self.aspect_ratio = aspect_ratio
        h_half_near = np.tan([hfov / 2]) * d_near
        self.near_left = -h_half_near[0]
        self.near_right = h_half_near[0]
        h_half_far = np.tan([hfov / 2]) * d_far
        self.far_left = -h_half_far[0]
        self.far_right = h_half_far[0]

procrustes_landmark_weight_map = {
    4: 0.070909939706326,
    6: 0.032100144773722,
    10: 0.008446550928056,
    33: 0.058724168688059,
    54: 0.007667080033571,
    67: 0.009078059345484,
    117: 0.009791937656701,
    119: 0.014565368182957,
    121: 0.018591361120343,
    127: 0.005197994410992,
    129: 0.120625205338001,
    132: 0.005560018587857,
    133: 0.05328618362546,
    136: 0.066890455782413,
    143: 0.014816547743976,
    147: 0.014262833632529,
    198: 0.025462191551924,
    205: 0.047252278774977,
    263: 0.058724168688059,
    284: 0.007667080033571,
    297: 0.009078059345484,
    346: 0.009791937656701,
    348: 0.014565368182957,
    350: 0.018591361120343,
    356: 0.005197994410992,
    358: 0.120625205338001,
    361: 0.005560018587857,
    362: 0.05328618362546,
    365: 0.066890455782413,
    372: 0.014816547743976,
    376: 0.014262833632529,
    420: 0.025462191551924,
    425: 0.047252278774977,
}

def build_anchors(anchor_num, input_map_shape, feature_map_size):
    anchors = []
    # get each shift by feature map size * anchor numbers
    for _y in range(0, feature_map_size, 1):
        for _x in range(0, feature_map_size, 1):
            temp_anch = Anchor(input_map_shape, feature_map_size, _x, _y)
            for n in range(0, anchor_num, 1):
                anchors.append(temp_anch)
    return anchors

def find_face_detection(raw_output, score, anchors, threshold):
  detected_points_list = []
  for j in range(0, len(anchors), 1):
      temp = raw_output[0, j, :]
      _score = score[0, j, 0]
      if _score <= threshold:
          continue
      anchor = anchors[j]
      box_ct_x = (anchor.center_x + temp[0])
      box_ct_y = (anchor.center_y + temp[1])
      box_w = temp[2]
      box_h = temp[3]
      _detected_pts = np.empty((9, 2))
      _detected_pts[0, k_dtct_w_idx] = box_ct_x - box_w / 2
      _detected_pts[0, k_dtct_h_idx] = box_ct_y - box_h / 2
      _detected_pts[1, k_dtct_w_idx] = box_ct_x + box_w / 2
      _detected_pts[1, k_dtct_h_idx] = box_ct_y + box_h / 2
      _detected_pts[2, k_dtct_w_idx] = box_ct_x
      _detected_pts[2, k_dtct_h_idx] = box_ct_y
      feature_index = 3
      for k in range(4, 6, 2):
          _detected_pts[feature_index, k_dtct_w_idx] = anchor.center_x + temp[k]
          _detected_pts[feature_index, k_dtct_h_idx] = anchor.center_y + temp[k + 1]
          feature_index = feature_index + 1
      detected_points_list.append(np.array([_detected_pts]))
  detected_points = np.concatenate(detected_points_list, axis = 0)
  return detected_points

def scale_detection(detection, input_w, input_h, target_shape):
    scale_w_input_to_target = target_shape[k_shp_w_idx] / input_w
    scale_h_input_to_target = target_shape[k_shp_h_idx] / input_h
    target_detection = np.empty(detection.shape)
    target_detection[:, :, k_dtct_w_idx] = \
        detection[:, :, k_dtct_w_idx] * scale_w_input_to_target
    target_detection[:, :, k_dtct_h_idx] = \
        detection[:, :, k_dtct_h_idx] * scale_h_input_to_target
    return target_detection

def scale_bbox(bbox, scale):
    scale_w = bbox.get_width() * scale
    scale_h = bbox.get_height() * scale
    (center_x, center_y) = bbox.get_center()
    scaled_bbox = Bbox(np.float32)
    scaled_bbox.tplft[k_dtct_w_idx] = center_x - scale_w / 2
    scaled_bbox.tplft[k_dtct_h_idx] = center_y - scale_w / 2
    scaled_bbox.btrht[k_dtct_w_idx] = center_x + scale_w / 2
    scaled_bbox.btrht[k_dtct_h_idx] = center_y + scale_w / 2
    return scaled_bbox

def normalize_bbox(bbox, base_w, base_h):
    n_bbox = Bbox(np.float32)
    n_bbox.tplft[k_dtct_w_idx] = bbox.tplft[k_dtct_w_idx] / base_w
    n_bbox.btrht[k_dtct_w_idx] = bbox.btrht[k_dtct_w_idx] / base_w
    n_bbox.tplft[k_dtct_h_idx] = bbox.tplft[k_dtct_h_idx] / base_h
    n_bbox.btrht[k_dtct_h_idx] = bbox.btrht[k_dtct_h_idx] / base_h
    return n_bbox

def get_cropped_image(image, bbox):
    int_bbox = Bbox(np.int32)
    int_bbox.tplft = bbox.tplft.astype(np.int32)
    int_bbox.btrht = bbox.btrht.astype(np.int32)
    print(int_bbox.tplft)
    print(int_bbox.btrht)
    return image[int_bbox.tplft[k_dtct_h_idx]:int_bbox.btrht[k_dtct_h_idx],
                 int_bbox.tplft[k_dtct_w_idx]:int_bbox.btrht[k_dtct_w_idx]]

def get_plane_lineset(arr_offset, scale, aspect_ratio = 1.0):

    points = np.array([
        [-0.5, 0.5 * aspect_ratio, 0],
        [0.5, 0.5 * aspect_ratio, 0],
        [0.5, -0.5 * aspect_ratio, 0],
        [-0.5, -0.5 * aspect_ratio, 0]
    ])
    points = points * scale
    points[:, 0] = points[:, 0] + arr_offset[0]
    points[:, 1] = points[:, 1] + arr_offset[1]
    points[:, 2] = points[:, 2] + arr_offset[2]
    lines = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def get_frustum_linesets(frustum):
    frustum_near_lines = get_plane_lineset([0, 0, frustum.d_near],
                                           frustum.near_right - frustum.near_left,
                                           frustum.aspect_ratio)
    frustum_far_lines = get_plane_lineset([0, 0, frustum.d_far],
                                          frustum.far_right - frustum.far_left,
                                          frustum.aspect_ratio)
    _points = np.asarray(frustum_near_lines.points)
    _points = np.concatenate([_points, np.asarray(frustum_far_lines.points)])
    _lines = np.array([
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(_points)
    line_set.lines = o3d.utility.Vector2iVector(_lines)
    colors = [[1, 0, 0] for i in range(len(_lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return (frustum_near_lines, frustum_far_lines, line_set)

def solve_weighted_orthogonal_problem_scale(source, target, weights):

    # jw (jw = Q * j)
    q_weights = np.sqrt(weights)
    q_weights = q_weights[:, np.newaxis]

    # transpose(Aw) (Aw = Q * A)
    weighted_source = np.transpose(np.multiply(source, q_weights))

    # transpose(Bw) (Bw = Q * B)
    weighted_target = np.transpose(np.multiply(target, q_weights))

    # w = transpose(jw) * jw
    total_weight = np.sum(np.multiply(q_weights, q_weights))

    # C = jw * transpose(jw) / transpose(jw) * jw = jw * transpose(jw) / w
    # source_centor_of_mass: c_w = (transpose(Aw) * jw / w)
    twice_weighted_source = np.multiply(weighted_source, np.transpose(q_weights))
    source_centor_of_mass = np.sum(twice_weighted_source, axis = 1) / total_weight

    # transpose(Aw) - transpose(Aw) * C = transpose(Aw) - c_w * transpose(jw)
    centered_weighted_source = weighted_source - np.matmul(source_centor_of_mass[:, np.newaxis], np.transpose(q_weights))

    # centered_weighted_source * Bw
    designed_matrix = np.matmul(weighted_target, np.transpose(centered_weighted_source))
    U, S, Vh = np.linalg.svd(designed_matrix, full_matrices=True)
    postrotation = U
    prerotation = Vh
    if np.linalg.det(postrotation) * np.linalg.det(prerotation) < 0.0:
        postrotation[:, 2] = postrotation[:, 2] * -1.0
    rotation = np.matmul(postrotation, prerotation)
    rotated_center_weighted_source = np.matmul(rotation, centered_weighted_source)
    numerator = np.sum(np.multiply(rotated_center_weighted_source, weighted_target))
    denominator = np.sum(np.multiply(centered_weighted_source, weighted_source))
    scale = numerator / denominator
    rotation_and_scale = scale * rotation
    pointwise_diffs = weighted_target - np.matmul(rotation_and_scale, weighted_source)
    weighted_pointwise_diffs = np.multiply(pointwise_diffs, np.transpose(q_weights))
    translation = np.sum(weighted_pointwise_diffs, axis = 1) / total_weight
    return (scale, rotation, translation)

def estimate_scale(cananical_face, detected_landmarks, weights):
    _temp_lmks = np.copy(detected_landmarks)
    #_temp_lmks[:, 2] = _temp_lmks[:, 2] * -1;  # change handiness
    scale, _discard, _discard = solve_weighted_orthogonal_problem_scale(
        cananical_face, _temp_lmks,  weights)
    return scale

def main():

    # part.1 face detection
    original_img = cv2.imread("./data/PXL_20231220_121718408.MP.jpg", cv2.IMREAD_UNCHANGED)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # blaze face detection (bf)
    # input 128 x 128 [-1, 1]
    bf_input_w = 128
    bf_input_h = 128
    bf_input = np.zeros([1, bf_input_w, bf_input_h, 3], dtype=np.float32)
    bf_input[0, :, :, :] = cv2.resize(original_img[:, :, 0:3],
                                      (bf_input_w, bf_input_h),
                                      cv2.INTER_CUBIC)
    # normalize to [-1, 1]
    bf_input = (bf_input / 127.5) - 1

    # prepare the tflite model
    bf_model_path = "./models/face_detector.tflite"

    # Load TFLite model and allocate tensors.
    bf_interpreter = tf.lite.Interpreter(model_path = bf_model_path)
    bf_interpreter.allocate_tensors()
    bf_input_details = bf_interpreter.get_input_details()
    bf_interpreter.set_tensor(bf_input_details[0]['index'], bf_input)

    ## Run inference
    bf_interpreter.invoke()
    # output_details[0]['index'] = the index which provides the input
    bf_output_details = bf_interpreter.get_output_details()
    bf_output = bf_interpreter.get_tensor(bf_output_details[0]['index'])
    bf_output_score = bf_interpreter.get_tensor(bf_output_details[1]['index'])

    # feature map size: 16 and 2 for each shift position
    # feature map size:  8 and 6 for each shift position
    anchors = build_anchors(2, bf_input.shape[1:3], 16)
    anchors.extend(build_anchors(6, bf_input.shape[1:3], 8))

    # find every detection
    # take the average of all the detections in stead of doing NMS
    bf_detection = find_face_detection(bf_output, bf_output_score, anchors, 0.75)
    mean_bf_detection = bf_detection.mean(axis = 0)

    # scale back the detecion into the original image size
    original_bf_detection = scale_detection(
        bf_detection, bf_input_w, bf_input_h, original_img.shape)
    mean_original_bf_detection = original_bf_detection.mean(axis = 0)

    # part.2 face landmark detection, and find the 3D scale of it

    # crop the face area on the original image as the input of the second model
    original_bbox = Bbox(np.float32)
    original_bbox.tplft = mean_original_bf_detection[0, :]
    original_bbox.btrht = mean_original_bf_detection[1, :]
    enlarged_ori_bbox = scale_bbox(original_bbox, 1.25)
    cropped_ori_img = get_cropped_image(original_img, enlarged_ori_bbox)
    #cv2.imshow("img0", cropped_ori_img)

    lm_input_w = 256
    lm_input_h = 256

    lm_input = np.zeros([1, lm_input_w, lm_input_h, 3], dtype=np.float32)
    lm_input[0, :, :, :] = cv2.resize(cropped_ori_img,
                                      (lm_input_w, lm_input_h),
                                      cv2.INTER_CUBIC)
    lm_input = (lm_input / 127.5) - 1

    lm_model_path = "./models/face_landmarks_detector.tflite"

    ## Load TFLite model and allocate tensors.
    lm_interpreter = tf.lite.Interpreter(model_path = lm_model_path)

    ## Allocate tensors
    lm_interpreter.allocate_tensors()

    #inference
    lm_input_details = lm_interpreter.get_input_details()
    lm_interpreter.set_tensor(lm_input_details[0]['index'], lm_input)
    lm_interpreter.invoke()

    lm_output_details = lm_interpreter.get_output_details()
    lm_output_id = lm_interpreter.get_tensor(lm_output_details[0]['index'])
    lm_output_id1 = lm_interpreter.get_tensor(lm_output_details[1]['index'])
    lm_output_id2 = lm_interpreter.get_tensor(lm_output_details[2]['index'])

    landmark_id_number = 478

    lm_id_3d = np.resize(lm_output_id[0, 0, 0, :], (landmark_id_number, 3))

    # get the normalized 3d points based on
    # the propotion from the cropped image to the original image
    lm_id_3d_normalized = np.empty(lm_id_3d.shape)
    lm_id_3d_normalized[:, 0] = lm_id_3d[:, 0] / lm_input_w
    lm_id_3d_normalized[:, 1] = lm_id_3d[:, 1] / lm_input_h
    lm_id_3d_normalized[:, 2] = lm_id_3d[:, 2] / lm_input_w

    normalized_bbox = normalize_bbox(enlarged_ori_bbox,
                                     original_img.shape[k_shp_w_idx],
                                     original_img.shape[k_shp_h_idx])

    _x_scale = normalized_bbox.get_width()
    _y_scale = normalized_bbox.get_height()
    _x_offset = normalized_bbox.tplft[k_dtct_w_idx]
    _y_offset = normalized_bbox.tplft[k_dtct_h_idx]
    lm_id_3d_normalized[:, 0] = lm_id_3d_normalized[:, 0] * _x_scale + _x_offset
    lm_id_3d_normalized[:, 1] = lm_id_3d_normalized[:, 1] * _y_scale + _y_offset
    lm_id_3d_normalized[:, 2] = lm_id_3d_normalized[:, 2] * _x_scale

    # set up a frutsum for the 3D scale calculation
    aspect_ratio = original_img.shape[k_shp_h_idx] / original_img.shape[k_shp_w_idx]
    frustum = Frustum(50 / 180.0 * np.pi, 1, 35, aspect_ratio)

    # scale the normalized 3d points into the size of the near plane of the frustum
    lm_id_3d_init = np.empty(lm_id_3d.shape)
    x_ratio = frustum.near_right - frustum.near_left
    y_ratio = x_ratio * frustum.aspect_ratio
    x_translation = frustum.near_left
    y_tranlation = frustum.near_left * frustum.aspect_ratio
    lm_id_3d_init[:, 0] = lm_id_3d_normalized[:, 0] * x_ratio + x_translation
    lm_id_3d_init[:, 1] = lm_id_3d_normalized[:, 1] * y_ratio + y_tranlation
    lm_id_3d_init[:, 2] = lm_id_3d_normalized[:, 2]

    # get the weightings
    landmark_weights = np.zeros(landmark_id_number)
    _key_filter_idx = []
    for key, val in procrustes_landmark_weight_map.items():
        landmark_weights[key] = val
        _key_filter_idx.append(key)
    _key_filter_idx = np.array(_key_filter_idx)

    # load canonical face can change its axis directions to
    # align with the image and face detection's axis definition
    canonical_face_obj = pywavefront.Wavefront('./data/face_model_with_iris.obj')
    canonical_face_points = np.asarray(canonical_face_obj.vertices)
    canonical_face_points[:, 1] = canonical_face_points[:, 1] * -1
    canonical_face_points[:, 2] = canonical_face_points[:, 2] * -1

    # estimate the scale from the canonical face size to
    # the size of the frustum near 3d points 
    first_scale = estimate_scale(canonical_face_points,
                                 lm_id_3d_init,
                                 landmark_weights)
    scale, rotation, translation = solve_weighted_orthogonal_problem_scale(
        canonical_face_points, lm_id_3d_init, landmark_weights)
    print("cananical_face to init scale:")
    print(scale)
    canonical_face_shrink_init_points = np.copy(canonical_face_points)
    for idx in range(0, landmark_id_number):
        canonical_face_shrink_init_points[idx, :] = np.matmul(
            scale *  rotation,
            (canonical_face_points[idx, :] - translation)[:, np.newaxis])[:, 0]

    # Move the z values to the frustum near plane
    lm_id_3d_near = np.copy(lm_id_3d_init)
    z_offset = np.mean(lm_id_3d_near[:, 2])
    lm_id_3d_near[:, 2] = lm_id_3d_near[:, 2] - z_offset + frustum.d_near
    canonical_face_shrink_init_points[:, 2] = canonical_face_shrink_init_points[:, 2] - z_offset + frustum.d_near

    lm_id_3d_step1 = np.copy(lm_id_3d_init)
    lm_id_3d_step1[:, 2] = np.divide(lm_id_3d_step1[:, 2] - z_offset + frustum.d_near, first_scale)
    lm_id_3d_step1[:, 0] = np.multiply(lm_id_3d_step1[:, 0], lm_id_3d_step1[:, 2]) / frustum.d_near
    lm_id_3d_step1[:, 1] = np.multiply(lm_id_3d_step1[:, 1], lm_id_3d_step1[:, 2]) / frustum.d_near

    second_scale = estimate_scale(canonical_face_points,
                                  lm_id_3d_step1,
                                  landmark_weights)
    total_scale = first_scale * second_scale
    lm_id_3d_step2 = np.copy(lm_id_3d_init)
    lm_id_3d_step2[:, 2] = np.divide(lm_id_3d_step2[:, 2] - z_offset + frustum.d_near, total_scale)
    lm_id_3d_step2[:, 0] = np.multiply(lm_id_3d_step2[:, 0], lm_id_3d_step2[:, 2]) / frustum.d_near
    lm_id_3d_step2[:, 1] = np.multiply(lm_id_3d_step2[:, 1], lm_id_3d_step2[:, 2]) / frustum.d_near

    lm_id_3d_step3 = np.copy(lm_id_3d_step2)
    lm_id_3d_step3_0 = np.copy(lm_id_3d_step3)
    scale, rotation, translation = solve_weighted_orthogonal_problem_scale(
        canonical_face_points, lm_id_3d_step3, landmark_weights)
    print("scale")
    print(scale)
    print("rotatino")
    print(rotation)
    print("translation")
    print(translation)
    for idx in range(0, len(landmark_weights)):
        lm_id_3d_step3[idx, :] = np.matmul(
            scale *  np.transpose(rotation),
            (lm_id_3d_step3_0[idx, :] - translation)[:, np.newaxis])[:, 0]

    canonical_face_pcd = o3d.geometry.PointCloud()
    canonical_face_pcd.points = o3d.utility.Vector3dVector(canonical_face_points)
    canonical_face_pcd.paint_uniform_color([0, 0.1, 1.0])

    canonical_face_shrink_init_pcd = o3d.geometry.PointCloud()
    canonical_face_shrink_init_pcd.points = o3d.utility.Vector3dVector(canonical_face_shrink_init_points)
    canonical_face_shrink_init_pcd.paint_uniform_color([0, 0.1, 1.0])

    canonical_face_weighted_pcd = o3d.geometry.PointCloud()
    canonical_face_weighted_pcd.points = o3d.utility.Vector3dVector(canonical_face_points[_key_filter_idx])
    canonical_face_weighted_pcd.paint_uniform_color([1.0, 0, 0])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0, origin = [0, 0, 0])

    lm_id_3d_normalized_pcd = o3d.geometry.PointCloud()
    lm_id_3d_normalized_pcd.points = o3d.utility.Vector3dVector(lm_id_3d_normalized)

    lm_id_3d_near_pcd = o3d.geometry.PointCloud()
    lm_id_3d_near_pcd.points = o3d.utility.Vector3dVector(lm_id_3d_near)

    lm_id_3d_step1_pcd = o3d.geometry.PointCloud()
    lm_id_3d_step1_pcd.points = o3d.utility.Vector3dVector(lm_id_3d_step1)
    lm_id_3d_step1_pcd.paint_uniform_color([1.0, 0.1, 0.0])

    lm_id_3d_step2_pcd = o3d.geometry.PointCloud()
    lm_id_3d_step2_pcd.points = o3d.utility.Vector3dVector(lm_id_3d_step2)
    lm_id_3d_step2_pcd.paint_uniform_color([1.0, 0.8, 0.0])

    lm_id_3d_step3_0_pcd = o3d.geometry.PointCloud()
    lm_id_3d_step3_0_pcd.points = o3d.utility.Vector3dVector(lm_id_3d_step3_0)
    lm_id_3d_step3_0_pcd.paint_uniform_color([0.0, 0.1, 0.0])

    lm_id_3d_step3_pcd = o3d.geometry.PointCloud()
    lm_id_3d_step3_pcd.points = o3d.utility.Vector3dVector(lm_id_3d_step3)
    lm_id_3d_step3_pcd.paint_uniform_color([0.0, 0.8, 0.0])

    o3d.visualization.draw([
                            mesh_frame,
                            get_plane_lineset(np.array([0.5, 0.5, 0]), 1.0),
                            *get_frustum_linesets(frustum),
                            lm_id_3d_normalized_pcd,
                            lm_id_3d_near_pcd,
                            canonical_face_pcd,
                            canonical_face_weighted_pcd,
                            canonical_face_shrink_init_pcd,
                            lm_id_3d_step1_pcd,
                            lm_id_3d_step2_pcd,
                            lm_id_3d_step3_0_pcd,
                            lm_id_3d_step3_pcd
                            ])

if __name__ == '__main__':
    main()

