from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class TransformConfig:
    prob: float = 0.175
    spacing: Optional[List[float]] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    orientation: str = "RAS"
    rand_bias_field: Optional[Dict[str, Any]] = None
    rand_gaussian_smooth: Optional[Dict[str, Any]] = None
    rand_gibbs_nose: Optional[Dict[str, Any]] = None
    rand_affine: Optional[Dict[str, Any]] = None
    rand_rotate90: Optional[Dict[str, Any]] = None
    rand_rotate: Optional[Dict[str, Any]] = None
    rand_elastic: Optional[Dict[str, Any]] = None
    rand_zoom: Optional[Dict[str, Any]] = None
    rand_crop_pos_neg_label: Optional[Dict[str, Any]] = None
    rand_spatial_crop_samples: Optional[Dict[str, Any]] = None
    gaussian_noise: Optional[Dict[str, Any]] = None
    shift_intensity: Optional[Dict[str, Any]] = None
    gaussian_sharpen: Optional[Dict[str, Any]] = None
    adjust_contrast: Optional[Dict[str, Any]] = None
    mode: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class DataConfig:
    image_cols: List[str] = field(default_factory=list)
    label_cols: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    out_channels: int


@dataclass
class Config:
    transforms: TransformConfig
    data: DataConfig
    model: ModelConfig
    ndim: int
    mode: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.transforms.mode = ('bilinear',) * len(self.data.image_cols) + \
                               ('nearest',) * len(self.data.label_cols)


def get_anatomy_transform_config() -> Config:
    return Config(
        transforms=TransformConfig(
            prob=0.175,
            spacing=[0.5, 0.5, 0.5],
            orientation="RAS",
            rand_bias_field={'degree': 10, 'coeff_range': [0.0, 0.01]},
            rand_gaussian_smooth={'sigma_x': [0.25, 1.5], 'sigma_y': [0.25, 1.5], 'sigma_z': [0.25, 1.5]},
            rand_gibbs_nose={'alpha': [0.5, 1]},
            rand_affine={'rotate_range': 5, 'shear_range': 0.5, 'translate_range': 25},
            rand_rotate90={'spatial_axes': [0, 1]},
            rand_rotate={'range_x': 0.1, 'range_y': 0.1, 'range_z': 0.1},
            rand_elastic={'sigma_range': [0.5, 1.5], 'magnitude_range': [0.5, 1.5], 'rotate_range': 5,
                          'shear_range': 0.5, 'translate_range': 25},
            rand_zoom={'min': 0.9, 'max': 1.1},
            # anatomy specific
            rand_crop_pos_neg_label={'spatial_size': [96, 96, 96], 'pos': 2, 'neg': 1, 'num_samples': 8},
            gaussian_noise={'mean': 0.1, 'std': 0.25},
            shift_intensity={'offsets': 0.2},
            gaussian_sharpen={'sigma1_x': [0.5, 1.0], 'sigma1_y': [0.5, 1.0], 'sigma1_z': [0.5, 1.0],
                              'sigma2_x': [0.5, 1.0], 'sigma2_y': [0.5, 1.0], 'sigma2_z': [0.5, 1.0],
                              'alpha': [10.0, 30.0]},
            adjust_contrast={'gamma': 2.0}
        ),
        data=DataConfig(
            image_cols=['t2'],  # anatomy, tumor [ 't2', 'adc', 'dwi' ]
            label_cols=['t2_anatomy_reader1']  # anatomy, tumor ['adc_tumor_reader1']
        ),
        model=ModelConfig(
            out_channels=3  # 3 for anatomy, 2 for tumor
        ),
        ndim=3,
        mode=['bilinear']
    )


def get_tumor_transform_config() -> Config:
    return Config(
        transforms=TransformConfig(
            prob=0.175,
            spacing=[0.5, 0.5, 0.5],
            orientation="RAS",
            rand_bias_field={'degree': 10, 'coeff_range': [0.0, 0.01]},
            rand_gaussian_smooth={'sigma_x': [0.25, 1.5], 'sigma_y': [0.25, 1.5], 'sigma_z': [0.25, 1.5]},
            rand_gibbs_nose={'alpha': [0.5, 1]},
            rand_affine={'rotate_range': 5, 'shear_range': 0.5, 'translate_range': 25},
            rand_rotate90={'spatial_axes': [0, 1]},
            rand_rotate={'range_x': 0.1, 'range_y': 0.1, 'range_z': 0.1},
            rand_elastic={'sigma_range': [0.5, 1.5], 'magnitude_range': [0.5, 1.5], 'rotate_range': 5,
                          'shear_range': 0.5, 'translate_range': 25},
            rand_zoom={'min': 0.9, 'max': 1.1},
            # tumor specific
            rand_spatial_crop_samples={'roi_size': [96, 96, 96], 'num_samples': 8},
            gaussian_noise={'mean': 0.1, 'std': 0.25},
            shift_intensity={'offsets': 0.2},
            gaussian_sharpen={'sigma1_x': [0.5, 1.0], 'sigma1_y': [0.5, 1.0], 'sigma1_z': [0.5, 1.0],
                              'sigma2_x': [0.5, 1.0], 'sigma2_y': [0.5, 1.0], 'sigma2_z': [0.5, 1.0],
                              'alpha': [10.0, 30.0]},
            adjust_contrast={'gamma': 2.0}
        ),
        data=DataConfig(
            image_cols=['t2', 'adc', 'dwi'],
            label_cols=['adc_tumor_reader1']
        ),
        model=ModelConfig(
            out_channels=2  # for tumor
        ),
        ndim=3,
        mode=['bilinear']
    )


def get_both_transform_config() -> Config:
    return Config(
        transforms=TransformConfig(
            prob=0.175,
            spacing=[0.5, 0.5, 0.5],
            orientation="RAS",
            rand_bias_field={'degree': 10, 'coeff_range': [0.0, 0.01]},
            rand_gaussian_smooth={'sigma_x': [0.25, 1.5], 'sigma_y': [0.25, 1.5], 'sigma_z': [0.25, 1.5]},
            rand_gibbs_nose={'alpha': [0.5, 1]},
            rand_affine={'rotate_range': 5, 'shear_range': 0.5, 'translate_range': 25},
            rand_rotate90={'spatial_axes': [0, 1]},
            rand_rotate={'range_x': 0.1, 'range_y': 0.1, 'range_z': 0.1},
            rand_elastic={'sigma_range': [0.5, 1.5], 'magnitude_range': [0.5, 1.5], 'rotate_range': 5,
                          'shear_range': 0.5, 'translate_range': 25},
            rand_zoom={'min': 0.9, 'max': 1.1},
            # tumor specific
            rand_spatial_crop_samples={'roi_size': [96, 96, 96], 'num_samples': 8},
            gaussian_noise={'mean': 0.1, 'std': 0.25},
            shift_intensity={'offsets': 0.2},
            gaussian_sharpen={'sigma1_x': [0.5, 1.0], 'sigma1_y': [0.5, 1.0], 'sigma1_z': [0.5, 1.0],
                              'sigma2_x': [0.5, 1.0], 'sigma2_y': [0.5, 1.0], 'sigma2_z': [0.5, 1.0],
                              'alpha': [10.0, 30.0]},
            adjust_contrast={'gamma': 2.0}
        ),
        data=DataConfig(
            image_cols=['t2', 'adc', 'dwi'],
            label_cols=['t2_anatomy_reader1', 'adc_tumor_reader1']
        ),
        model=ModelConfig(
            out_channels=4  # for 1 bg + 2 anatomy + 1 tumor
        ),
        ndim=3,
        mode=['bilinear']
    )


if __name__ == '__main__':
    # Example usage
    config = Config(
        transforms=TransformConfig(
            prob=0.175,
            spacing=[0.5, 0.5, 0.5],
            orientation="RAS",
            rand_bias_field={'degree': 10, 'coeff_range': [0.0, 0.01]},
            rand_gaussian_smooth={'sigma_x': [0.25, 1.5], 'sigma_y': [0.25, 1.5], 'sigma_z': [0.25, 1.5]},
            rand_gibbs_nose={'alpha': [0.5, 1]},
            rand_affine={'rotate_range': 5, 'shear_range': 0.5, 'translate_range': 25},
            rand_rotate90={'spatial_axes': [0, 1]},
            rand_rotate={'range_x': 0.1, 'range_y': 0.1, 'range_z': 0.1},
            rand_elastic={'sigma_range': [0.5, 1.5], 'magnitude_range': [0.5, 1.5], 'rotate_range': 5,
                          'shear_range': 0.5, 'translate_range': 25},
            rand_zoom={'min': 0.9, 'max': 1.1},
            # anatomy specific
            rand_crop_pos_neg_label={'spatial_size': [96, 96, 96], 'pos': 2, 'neg': 1, 'num_samples': 8},
            # tumor specific
            rand_spatial_crop_samples={'roi_size': [96, 96, 96], 'num_samples': 8},
            gaussian_noise={'mean': 0.1, 'std': 0.25},
            shift_intensity={'offsets': 0.2},
            gaussian_sharpen={'sigma1_x': [0.5, 1.0], 'sigma1_y': [0.5, 1.0], 'sigma1_z': [0.5, 1.0],
                              'sigma2_x': [0.5, 1.0], 'sigma2_y': [0.5, 1.0], 'sigma2_z': [0.5, 1.0],
                              'alpha': [10.0, 30.0]},
            adjust_contrast={'gamma': 2.0}
        ),
        data=DataConfig(
            image_cols=['t2'],  # anatomy, tumor [ 't2', 'adc', 'dwi' ]
            label_cols=['t2_anatomy_reader1']  # anatomy, tumor ['adc_tumor_reader1']
        ),
        model=ModelConfig(
            out_channels=3  # 3 for anatomy, 2 for tumor
        ),
        ndim=3,
        mode=['bilinear']
    )
