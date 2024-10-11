import torch

def convert_multiclass(
    checkpoint_path='pretrain/multi_class_model.pth',
    output_path='pretrain/multi_class_model_with_classes.pth',
    classes=['wrecks/ruins', 'fish', 'reefs', 'aquatic plants', 'human divers', 'robots', 'sea-floor']
):
    """
    Function to load a PyTorch model checkpoint, add class metadata, and save the modified checkpoint.

    Args:
    checkpoint_path (str): Path to the input model checkpoint (default: 'pretrain/multi_class_model.pth').
    output_path (str): Path to save the modified checkpoint (default: 'pretrain/multi_class_model_with_classes.pth').
    classes (list): List of class names to add to the checkpoint metadata (default: list of marine-related classes).
    """
    # Load the checkpoint using PyTorch
    checkpoint = torch.load(checkpoint_path)
    
    # Ensure 'meta' field exists in the checkpoint
    if 'meta' not in checkpoint:
        checkpoint['meta'] = {}
    
    # Add class information to the checkpoint metadata
    checkpoint['meta']['dataset_meta'] = dict(classes=classes)
    
    # Save the modified checkpoint
    torch.save(checkpoint, output_path)


if __name__ == '__main__':
    convert_multiclass()