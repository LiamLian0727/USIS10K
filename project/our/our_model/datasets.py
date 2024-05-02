from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class ForegroundUSIS10KInsSegDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ['foreground'],
        'palette': [(0, 0, 255)]
    }


@DATASETS.register_module()
class MultiClassUSIS10KInsSegDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants',
                    'human divers', 'robots', 'sea-floor'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230)]
    }
