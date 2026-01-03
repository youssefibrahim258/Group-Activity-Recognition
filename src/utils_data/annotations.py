
def load_video_annotations(annotation_file):
    """
        Load video annotations.

    Args:
        annotation_file (str): Path to annotation file.

    Returns:
        dict: Mapping from clip/frame name to label.
    """
    clip_to_label = {}

    with open(annotation_file, "r") as f:
        for line in f:
            items = line.strip().split()
            clip = items[0].replace(".jpg", "")
            label = items[1]
            clip_to_label[clip] = label

    return clip_to_label
