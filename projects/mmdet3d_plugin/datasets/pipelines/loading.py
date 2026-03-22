import numpy as np
import cv2
import mmcv
from mmdet.datasets.builder import PIPELINES
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        results["points"] = points
        return results

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_omnihd(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    # ----PhotoMetricDistortionMultiViewImage需要to_float32-----
    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views) （1080,1920,3,6）,前后视是3840*2160,读进来先去畸变-
        # -----------这里读取6路图像，尺寸不一样大可以参考后面scale操作----------
        # ----------在这里先scale掉前后视，lidar2img和原始内参也要对应改-------------
        img = []
        lidar2img = []
        cam_intrinsic = []
        scale_factor_frontandback = 0.5
        for i, name in enumerate(filename):
            current_img = mmcv.imread(name, self.color_type)
            # cv2.imwrite(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/saved_image_ori_{i}.jpg', current_img)
            current_img_undistortion = cv2.undistort(current_img, results['cam_intrinsic'][i][:3, :3], \
                                                     results['cam_distortion'][i], None,
                                                     results['cam_intrinsic'][i][:3, :3])
            # cv2.imwrite(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/saved_image_dist_{i}.jpg', current_img_undistortion)
            if name.split('/')[-2] == 'camera_front' or name.split('/')[-2] == 'camera_back':
                img_final = mmcv.imresize(current_img_undistortion, (
                int(current_img_undistortion.shape[1] * scale_factor_frontandback),
                int(current_img_undistortion.shape[0] * scale_factor_frontandback)), return_scale=False)
                # cv2.imwrite(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/saved_image_resize_{i}.jpg', img_final)
                img.append(img_final)
                scale_factor = np.eye(4)
                scale_factor[0, 0] *= scale_factor_frontandback
                scale_factor[1, 1] *= scale_factor_frontandback
                current_lidar2img = scale_factor @ results['lidar2img'][i]
                cam_intrinsic.append(scale_factor @ results['cam_intrinsic'][i])
                lidar2img.append(current_lidar2img)
            else:
                img.append(current_img_undistortion)
                lidar2img.append(results['lidar2img'][i])
                cam_intrinsic.append(results['cam_intrinsic'][i])

        img = np.stack(img, axis=-1)  # ----（1080,1920,3,6）---
        results['lidar2img'] = lidar2img
        results['cam_intrinsic'] = cam_intrinsic
        if self.to_float32:
            img = img.astype(np.float32)  # --float32--
        results['filename'] = filename  # ---赋值filename字段---
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]  # --按顺序六个图像，(1080,1920,3)--
        results['img_shape'] = img.shape  # ---下面都是赋初始值字段----
        results['ori_shape'] = img.shape  # ----（1080,1920,3,6）
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape  # ----（1080,1920,3,6）
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)  # ------这里初始值是mean0,std1-------

        # #------------debug真值投影----------

        # from .draw_box_on_img import draw_lidar_bbox3d_on_img
        # for i in range(6):
        #     img = results['img'][i]
        #     img = img.copy()
        #     img = draw_lidar_bbox3d_on_img(results['gt_bboxes_3d'], img, results['lidar2img'][i], None)
        #     cv2.imwrite(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_box_debug/gt_bbox_to_img_{i}.jpg', img)
        # #------------------------------------
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


# --原始[x,y,z,velocity,power,motion_state,SNR,valid_flag]----
# --生成的格式x y z vx_r_comp vy_r_comp, power,snr,time_diff,Vr[每个radar坐标下],radar_ID
# -使用x y z vx_r_comp vy_r_comp, power,snr,time_diff补偿的速度，旋转到ego/lidar坐标系
@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps_omnihd(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=8,
                 use_dim=[0, 1, 2, 3, 4, 5, 6, 7],
                 sweeps_num=3,
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-72, -56, -3.0, 72, 56, 5.0],
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range

    # --------读取radar点云，bin文件,8维------------------
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def _pad_or_drop(self, points):
        '''
        points: [N, 8]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1),
                            dtype=points.dtype)

            return points, masks

        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1),
                            dtype=points.dtype)

            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]),
                             dtype=points.dtype)
            masks = np.ones((num_points, 1),
                            dtype=points.dtype)

            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radars']
        radar_ID = {'radar_front': 0,
                    'radar_left_front': 1,
                    'radar_right_front': 2,
                    'radar_back': 3,
                    'radar_left_back': 4,
                    'radar_right_back': 5}

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))

            ts = int(sweeps[0]['timestamp']) * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = int(sweep['timestamp']) * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # ---------------进行速度补偿并只用补偿后的速度，转到当前帧lidar/ego坐标系-------------
                # -------------------相对径向速度和补偿后的速度投影到当前帧坐标系--------------------------
                # velocity compensated by the ego motion in sensor frame
                # 提取 xyz 坐标和径向速度 vr
                xyz = points_sweep[:, :3]
                vr = points_sweep[:, 3]
                # 计算点的距离 r
                r = np.linalg.norm(xyz, axis=1)

                # 计算方位角 azimuth
                azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])  # ---- 前后视+-70/+-10

                # 计算仰角 elevation
                elevation = np.arcsin(xyz[:, 2] / r)  # --前后视+-12/+-5

                # 自车速度在radar方向分解
                V_ego = np.array(sweep['ego_velocity']).reshape(-1, 3)

                V_sensor = V_ego @ np.linalg.inv(Quaternion(sweep['sensor2ego_rotation']).rotation_matrix).T
                V_sensor = np.repeat(V_sensor, points_sweep.shape[0], axis=0)
                # 计算自车传感器速度在径向速度 vr 方向上的分量进行速度补偿
                Vr_compensated = V_sensor[:, 0] * np.cos(azimuth) * np.cos(elevation) + \
                                 V_sensor[:, 1] * np.sin(azimuth) * np.cos(elevation) + \
                                 V_sensor[:, 2] * np.sin(elevation) + vr
                # 计算补偿后的x方向速度 Vx_compensated
                Vx_compensated = Vr_compensated * np.cos(elevation) * np.cos(azimuth)
                # 计算补偿后的y方向速度 Vy_compensated
                Vy_compensated = Vr_compensated * np.cos(elevation) * np.sin(azimuth)

                velo_comp = np.concatenate((Vx_compensated.reshape(-1, 1), Vy_compensated.reshape(-1, 1)), axis=1)
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T  # ---这里转到了当前帧lidar坐标系
                velo_comp = velo_comp[:, :2]

                # ----------------------------点云转到当前坐标系--------------------------------
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']

                points_radarID = np.full((points_sweep.shape[0], 1), radar_ID[key])
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :3],
                     velo_comp, points_sweep[:, [4, 6]],
                     time_diff, Vr_compensated.reshape(-1, 1), points_radarID],
                    axis=1)  # -使用x y z vx_r_comp vy_r_comp, power,snr,time_diff,Vr[每个radar坐标下],radar_ID
                points_sweep_list.append(points_sweep_)

        points = np.concatenate(points_sweep_list, axis=0)  # ---8000-20000

        points = points[:, self.use_dim]

        # -------------PointsRangeFilter放在这里，过滤范围外的点云------------------
        radar_mask = ((points[:, 0] > self.pc_range[0])
                      & (points[:, 1] > self.pc_range[1])
                      & (points[:, 2] > self.pc_range[2])
                      & (points[:, 0] < self.pc_range[3])
                      & (points[:, 1] < self.pc_range[4])
                      & (points[:, 2] < self.pc_range[5]))
        clean_radar = points[radar_mask]

        results['points'] = clean_radar.astype(np.float32)  # ---这里写成points----

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3,
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 test_mode=False,
                 rote90=True,
                 ignore=[]):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.ignore = ignore
        self.rote90 = rote90
        if len(ignore) > 0:
            print(self.ignore)

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        radar_obj = RadarPointCloud.from_file(pts_filename)

        # [18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1),
                            dtype=points.dtype)

            return points, masks

        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1),
                            dtype=points.dtype)

            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]),
                             dtype=points.dtype)
            masks = np.ones((num_points, 1),
                            dtype=points.dtype)

            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radars']
        # print(radars_dict.keys())

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if key in self.ignore:
                continue
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))

            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                # print(time_diff)
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # print()
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)
                points_sweep_list.append(points_sweep_)

        points = np.concatenate(points_sweep_list, axis=0)
        # print(points.shape)

        points = points[:, self.use_dim]

        # print(points.shape[-1])

        if (self.rote90):
            # rotate -90 degree to align with lidar coordinate
            # points.rotate(-math.pi / 2)
            rot_sin = np.sin( -np.pi / 2)
            rot_cos = np.cos( -np.pi / 2)
            rot_mat_T = np.array([
                [rot_cos, -rot_sin, 0],
                [rot_sin, rot_cos, 0],
                [0, 0, 1]
            ], dtype=np.float32).T
            points[:, :3] = points[:, :3] @ rot_mat_T
            points[:, 3:5] = points[:, 3:5] @ rot_mat_T[:2, :2]  # only rotate x,y velocity

        # -------------PointsRangeFilter放在这里，过滤范围外的点云------------------
        radar_mask = ((points[:, 0] > self.pc_range[0])
                      & (points[:, 1] > self.pc_range[1])
                      & (points[:, 2] > self.pc_range[2])
                      & (points[:, 0] < self.pc_range[3])
                      & (points[:, 1] < self.pc_range[4])
                      & (points[:, 2] < self.pc_range[5]))
        clean_radar = points[radar_mask]

        results['points'] = clean_radar.astype(np.float32)  # ---这里写成points----

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'
