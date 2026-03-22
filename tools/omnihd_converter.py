from typing import List
import os
import os.path as osp
import numpy as np
from pyquaternion import Quaternion
import mmcv

from newscenes_devkit.newscenes import NewScenes

NameMapping = {
    "suv": "car",
    "van": "car",
    "truck": "large_vehicle",
    "rider": "rider",
    # 'cyclist':'rider',#----原来翻译成了cyclist这里兼容一下防止漏掉
    "pedestrian": "pedestrian",
    "car": "car",
    "tricyclist": "car",
    "light_truck": "large_vehicle",
    "bus": "large_vehicle",
    "engineering_vehicle": "large_vehicle",
    "handcart": "car",
    "trailer": "large_vehicle",
}  # ---与category_to_detection_name保持一致


def RT_transform_to_quaternion(RT_matrix: List[float]) -> List[float]:
    """Convert the RT matrix to quaternion.
    """
    transformation_matrix = np.array(RT_matrix)
    if transformation_matrix.shape != (4, 4):
        transformation_matrix = transformation_matrix.reshape(4, 4)

    quaternion = Quaternion(matrix=transformation_matrix[:3, :3], atol=1e-4)
    rotation = quaternion.elements.astype(float).tolist()
    translation = transformation_matrix[:3, 3].astype(float).tolist()
    return rotation, translation


def _get_can_bus_info(lidar_pose_record, canbus_record):
    rotation, translation = RT_transform_to_quaternion(lidar_pose_record['pose'])
    acc_xyz = canbus_record['acc_xyz']
    gyro_xyz = canbus_record['gyro_xyz']
    velocity_ego = canbus_record['velocity_ego']

    can_bus = translation + rotation + acc_xyz + gyro_xyz + velocity_ego + [0., 0.]

    return np.array(can_bus)


def obtain_sensor2top(newsc,
                      sync_token,
                      scene_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        newsc (class): Dataset class in the newScenes dataset.
        sync_token (str): sync token corresponding to the
            sync file.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sync_record = newsc.get('sample_data', sync_token)

    if sensor_type == 'lidar':
        data_path = osp.join(newsc.dataroot, sync_record['lidar']['lidar_top_compensation'])
        cs_record = {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]}
        ego_pose_token = sync_record['ego_pose']['lidar_top_compensation']

        sensor_timestamp = sync_token



    elif sensor_type[:3] == 'cam':
        data_path = osp.join(newsc.dataroot, sync_record['cameras'][sensor_type])
        RT_matrix = newsc.get('sensor_calibration', scene_token)['calib'][sensor_type]['camera2ego']
        rotation_cam, translation_cam = RT_transform_to_quaternion(RT_matrix)
        cs_record = {'translation': translation_cam,
                     'rotation': rotation_cam}
        ego_pose_token = sync_record['ego_pose']['lidar_top_compensation']

        sensor_timestamp = sync_token


    elif sensor_type[:3] == 'rad':
        data_path = osp.join(newsc.dataroot, sync_record['radars'][sensor_type])
        RT_matrix = newsc.get('sensor_calibration', scene_token)['calib'][sensor_type]['radar2ego']
        rotation_rad, translation_rad = RT_transform_to_quaternion(RT_matrix)
        cs_record = {'translation': translation_rad,
                     'rotation': rotation_rad}
        ego_pose_token = sync_record['ego_pose'][sensor_type]
        sensor_timestamp = sync_record['radars'][sensor_type].split('/')[-1][:-4]

    else:
        raise ValueError('unknown sensor type')

    pose_matrix = newsc.get('ego_pose', ego_pose_token, scene_token)['pose']
    canbus_record = newsc.get('imu_data', ego_pose_token, scene_token)
    rotation_pose, translation_pose = RT_transform_to_quaternion(pose_matrix)

    pose_record = {'translation': translation_pose,
                   'rotation': rotation_pose}
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sync_token,
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'ego_velocity': canbus_record['velocity_ego'],
        'timestamp': sensor_timestamp
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # 这里计算过程是(e2l*g2e*e2g*s2e).T.T
    sweep['sensor2lidar_translation'] = T
    return sweep


def _fill_trainval_infos(omni, train_scenes, val_scenes, test, max_sweeps):
    train_omni_infos = []
    val_omni_infos = []

    for sample in mmcv.track_iter_progress(omni.sample):
        lidar_token = sample['token']
        sync_record = omni.get('sample_data', sample['token'])

        lidar_pose_record = omni.get('ego_pose', sync_record['ego_pose']['lidar_top_compensation'],
                                     sample['scene_token'])
        canbus_record = omni.get('imu_data', sync_record['ego_pose']['lidar_top_compensation'], sample['scene_token'])

        lidar_path = osp.join(omni.dataroot, sync_record['lidar']['lidar_top_compensation'])
        boxes = omni.get_annotation_box(lidar_token)

        mmcv.check_file_exist(lidar_path)

        can_bus = _get_can_bus_info(lidar_pose_record, canbus_record)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],  # additional
            'next': sample['next'],  # additional
            'can_bus': can_bus,  # additional
            'frame_idx': sample['frame_idx'],  # additional temporal related info
            'sweeps': [],
            'cams': dict(),
            'radars': dict(),  # additional
            'scene_token': sample['scene_token'],  # additional temporal related info
            'lidar2ego_translation': [0.0, 0.0, 0.0],
            'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
            'ego2global_translation': can_bus[:3],
            'ego2global_rotation': can_bus[3:7],
            'timestamp': int(sample['timestamp']),
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 camera's information per frame
        camera_types = [
            'camera_front',
            'camera_left_front',
            'camera_right_front',
            'camera_back',
            'camera_left_back',
            'camera_right_back'
        ]
        for cam in camera_types:
            cam_intrinsic = omni.get('sensor_calibration', info['scene_token'])['calib'][cam]['intrinsic']
            cam_distortion = omni.get('sensor_calibration', info['scene_token'])['calib'][cam]['distortion']

            # additional key: ego_velocity
            cam_info = obtain_sensor2top(omni, sync_record['token'], info['scene_token'], l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            cam_info.update(cam_distortion=cam_distortion)
            info["cams"].update({cam: cam_info})

        # obtain 6 radar's information per frame
        radar_types = [
            'radar_front',
            'radar_left_front',
            'radar_right_front',
            'radar_back',
            'radar_left_back',
            'radar_right_back'
        ]
        for radar_name in radar_types:
            radar_token = sync_record['token']
            sweeps = []
            while len(sweeps) < 3:
                if not omni.get('sample_data', radar_token)['prev'] == '':
                    radar_info = obtain_sensor2top(omni, radar_token, info['scene_token'], l2e_t, l2e_r_mat,
                                                   e2g_t, e2g_r_mat, radar_name)
                    sweeps.append(radar_info)
                    radar_token = omni.get('sample_data', radar_token)['prev']  # ---token赋值成前一帧sync
                else:
                    radar_info = obtain_sensor2top(omni, radar_token, info['scene_token'], l2e_t, l2e_r_mat,
                                                   e2g_t, e2g_r_mat, radar_name)
                    sweeps.append(radar_info)
            info['radars'].update({radar_name: sweeps})

        # obtain sweeps for a single key-frame
        lidar_sweeps = []
        lidar_token = sync_record['token']
        while len(lidar_sweeps) < max_sweeps:
            if not omni.get('sample_data', lidar_token)['prev'] == '':
                sweep = obtain_sensor2top(omni, omni.get('sample_data', lidar_token)['prev'], info['scene_token'],
                                          l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                lidar_sweeps.append(sweep)
                lidar_token = omni.get('sample_data', lidar_token)['prev']
            else:
                break
        info['sweeps'] = lidar_sweeps

        # obtain annotation
        if not test:
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([b.velocity[:2] for b in boxes]).reshape(-1, 2)

            track_id = np.array([b.track_id for b in boxes])

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
                else:
                    print(f'不存在此类{names[i]}')
                    return
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)  # not use LiDARInstance3DBoxes
            assert len(gt_boxes) == len(boxes), f'{len(gt_boxes)}, {len(boxes)}'

            info["instance_inds"] = track_id  # use track_id as instance index
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.full(len(gt_boxes), -1, dtype=int)  # unused
            info['num_radar_pts'] = np.full(len(gt_boxes), -1, dtype=int)  # unused
            visibility = np.array([b.visibility for b in boxes], dtype=bool)
            info['visibility'] = visibility  # additional
            info['valid_flag'] = visibility

        if sample['scene_token'] in train_scenes:
            train_omni_infos.append(info)
        if sample['scene_token'] in val_scenes:
            val_omni_infos.append(info)

    return train_omni_infos, val_omni_infos


def create_omnihd_infos(root_path, info_prefix, version, max_sweeps):
    omni = NewScenes(version=version, dataroot=root_path, verbose=True)

    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = omni.scene_split['train']
        val_scenes = omni.scene_split['val']
    elif version == 'v1.0-test':
        train_scenes = omni.scene_split['test']
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = omni.scene_split['train_mini']
        val_scenes = omni.scene_split['val_mini']
    else:
        raise ValueError('unknown')

    # -----------if existing scenes.-------------------------
    all_scenes = sorted(next(os.walk(omni.dataroot))[1])
    if all(scene in all_scenes for scene in train_scenes) and \
            val_scenes != [] and all(scene in all_scenes for scene in val_scenes):
        print('Train and Val scenes exist.')
    elif all(scene in all_scenes for scene in train_scenes) and val_scenes == []:
        print('Test scenes exist.')
    else:
        raise ValueError('Some scenes do not exist.')
    # --------------------------------------------

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(len(train_scenes), len(val_scenes)))

    train_omni_infos, val_omni_infos = _fill_trainval_infos(
        omni, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_omni_infos)))
        data = dict(infos=train_omni_infos, metadata=metadata)
        info_path = "{}_infos_test.pkl".format(info_prefix)
        mmcv.dump(data, info_path)
    else:
        print("train sample: {}, val sample: {}".format(len(train_omni_infos), len(val_omni_infos)))
        data = dict(infos=train_omni_infos, metadata=metadata)
        info_path = "{}_infos_train.pkl".format(info_prefix)
        mmcv.dump(data, info_path)
        data["infos"] = val_omni_infos
        info_val_path = "{}_infos_val.pkl".format(info_prefix)
        mmcv.dump(data, info_val_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="omnihd converter")
    parser.add_argument("--root_path", type=str, default="./data/omnihd")
    parser.add_argument("--info_prefix", type=str, default="omnihd")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--max_sweeps", type=int, default=2)
    args = parser.parse_args()

    create_omnihd_infos(
        args.root_path,
        args.info_prefix,
        version=args.version,
        max_sweeps=args.max_sweeps,
    )
