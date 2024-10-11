# This script is used to visualize the prediction results
from mmdet.apis import DetInferencer

def vis_infer(
    checkpoints="./pretrain/multi_class_model_with_classes.pth",
    config='./project/our/configs/multiclass_usis_train.py',
    data_dir='./data/USIS10K/test/',
    output_dir='./USIS10K/data/vis/test'
):
    """
    Function to run the DetInferencer for visual inference with default parameters.

    Args:
    checkpoints (str): Path to the model checkpoint (default: "./pretrain/multi_class_model_with_classes.pth").
    config (str): Path to the configuration file (default: './project/our/configs/multiclass_usis_train.py').
    data_dir (str): Path to the directory containing the test data (default: './data/USIS10K/test/').
    output_dir (str): Path to the output directory where results will be saved (default: './USIS10K/data/vis/test').
    """
    # Initialize the DetInferencer
    inferencer = DetInferencer(model=config, weights=checkpoints)
    
    # Perform inference and save the output
    inferencer(data_dir, out_dir=output_dir)


if __name__ == "__main__":
    vis_infer()
