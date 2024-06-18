from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CamVid(BaseSegDataset):
    METAINFO = dict(
        classes=('Bicyclist', 'Building', 'Car', 'Column_Pole',
                 'Fence', 'Pedestrian', 'Road', 'Sidewalk',
                 'SignSymbol', 'Sky', 'Tree'),
        palette=[[0, 128, 192], [128, 0, 0], [64, 0, 128], [192, 192, 128],
                 [64, 64, 128], [64, 64, 0], [128, 64, 128], [0, 0, 192],
                 [192, 128, 128], [128, 128, 128], [128, 128, 0]]
    )

    def __init__(self, **kwargs):
        super(CamVid, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_L.png',
            reduce_zero_label=False,
            **kwargs)
