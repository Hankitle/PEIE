import cv2
import numpy as np
import random
from torch.utils import data as data

from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from .data_util import paths_from_folder
from .transforms import augment, list_random_crop, list_resize
from .degradation import add_blur, add_Gaussian_noise, add_JPEG_noise
from .opencv_randaugment import OpenCVAugment
    
@DATASET_REGISTRY.register()
class PEIEDataset(data.Dataset):
    """Dataset used for PEIE model:
    PEIE: Physics Embedded Illumination Estimation for Adaptive Dehazing.

    It loads gt (Ground-Truth) images, and augments them.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_depth (str): Data root path for depth.
            dataroot_hq (str): Data root path for hq.

            beta_range : [0.8, 2.0]
            A_range : [0.8, 1.0]
            color_range : [-0.2, 0]

            train_size (int)
            use_resize_crop (bool)
            use_hflip (bool): Use horizontal flips.
            # Please see more options in the codes.
    """

    def __init__(self, opt):
        super(PEIEDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.depth_folder = opt['dataroot_depth']
        self.hq_folder = opt['dataroot_hq']

        self.gt_paths = paths_from_folder(self.gt_folder)
        self.depth_paths = paths_from_folder(self.depth_folder)
        self.hq_paths = paths_from_folder(self.hq_folder)

        self.light_range = opt['light_range']
        self.light_probabilities = opt['light_probabilities']
        self.beta_range = opt['beta_range']
        self.A_range = opt['A_range']
        self.color_range = opt['color_range']

        if opt.get('train_size') is not None:
            self.train_size = opt['train_size']

        self.in_camera_degradation = OpenCVAugment(2)

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        depth_path = self.depth_paths[index]
        hq_path = self.hq_paths[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # incident illumination adjustment
        img_hq = cv2.imread(hq_path).astype(np.float32) / 255.0
        LI = (img_gt / (img_hq + 0.15)).clip(0.01, 1)
        gamma = self._random_in_range(self.light_range, self.light_probabilities)
        img_lq = ((np.power(LI, gamma) * img_hq)).clip(0, 1)
            
        # add haze
        A = self._random_in_range(self.A_range) - self._random_in_range(self.color_range, dim=3)
        beta = self._random_in_range(self.beta_range)

        # depth refinement
        depth = 1 - depth
        t = np.exp(-beta * depth)[..., np.newaxis]
        img_lq = img_lq * t + A * (1 - t)

        # resize then crop
        if self.opt['use_resize_crop']:
            min_dim = min(img_gt.shape[:2])
            new_size = random.randint(self.train_size, self.train_size + 32) if min_dim >= (self.train_size + 32) else self.train_size
            resize_factor = new_size / min_dim

            img_lq, img_hq = list_resize([img_lq, img_hq], resize_factor)
            img_lq, img_hq = list_random_crop([img_lq, img_hq], self.train_size)
        img_lq, img_hq = augment([img_lq, img_hq], hflip=self.opt['use_hflip'], rotation=False)

        # environmental illumination simulation
        if np.random.rand() < 0.2:
            img_lq = add_blur(img_lq)

        # in-camera degradation
        img_lq = self.in_camera_degradation(img_lq)

        img_hq = img_hq.clip(0, 1)[..., :3]
        img_lq = img_lq.clip(0, 1)[..., :3]
            
        img_hq, img_lq = img2tensor([img_hq, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_hq,
            'gt_path': gt_path,
            'syn_params': {
                'A': A,
                'beta': beta,
                'gamma': gamma
            }
        }

    def __len__(self):
        return len(self.gt_paths)

    def update_train_size(self, train_size):
        self.train_size = train_size

    def _random_in_range(self, boundaries, probabilities=[1.], dim=1):
        assert len(boundaries) - 1 == len(probabilities), "boundaries must be one more than probabilities"
        assert np.isclose(sum(probabilities), 1), "probabilities must sum to 1"
        boundaries = np.array(boundaries, dtype=np.float32)

        if isinstance(dim, int) and dim > 1:
            chosen_intervals = np.random.choice(len(probabilities), size=dim, p=probabilities)
            interval_starts = boundaries[chosen_intervals]
            interval_ends = boundaries[chosen_intervals + 1]

            return np.random.uniform(interval_starts, interval_ends).astype(np.float32)
        else:
            chosen_interval = np.random.choice(len(probabilities), p=probabilities)
            interval_start = boundaries[chosen_interval]
            interval_end = boundaries[chosen_interval + 1]

            return np.random.uniform(interval_start, interval_end)
