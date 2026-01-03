import os
from PIL import Image
from torch.utils.data import Dataset
from src.utils_data.annotations import load_video_annotations


class VolleyballClipDataset(Dataset):
    def __init__(self, videos_dir, video_ids, label_encoder, transform=None):

        self.samples = []
        self.transform = transform
        self.label_encoder = label_encoder

        for vid in video_ids:
            video_path = os.path.join(videos_dir, str(vid))
            annot_path = os.path.join(video_path, "annotations.txt")

            clip_labels = load_video_annotations(annot_path)

            for clip_name, label_str in clip_labels.items():
                clip_dir = os.path.join(video_path, clip_name)

                if not os.path.exists(clip_dir):
                    continue

                frames = sorted(os.listdir(clip_dir))
                if len(frames) == 0:
                    continue

                mid_idx = len(frames) // 2
                frame_path = os.path.join(clip_dir, frames[mid_idx])

                self.samples.append((frame_path, label_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        try:
            label_id = self.label_encoder.encode(label_str)
        except KeyError:
            raise ValueError(
                f"Label '{label_str}' not found in LabelEncoder classes")

        return image, label_id
