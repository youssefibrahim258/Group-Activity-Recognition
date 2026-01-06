import os
from PIL import Image
from torch.utils.data import Dataset
from src.utils_data.annotations import load_video_annotations


class VolleyballClip9FramesDataset(Dataset):
    def __init__(
        self,
        videos_dir,
        video_ids,
        label_encoder,
        transform=None,
        num_frames_before=4,
        num_frames_after=4,
        repeat=1
    ):
        """
        Each clip -> 9 independent frame samples (non-temporal)

        Sample = single frame (C, H, W)
        """
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
                start_idx = max(0, mid_idx - num_frames_before)
                end_idx = min(len(frames), mid_idx + num_frames_after + 1)

                selected_frames = frames[start_idx:end_idx]

                # كل فريم = sample مستقل
                for frame_name in selected_frames:
                    frame_path = os.path.join(clip_dir, frame_name)
                    for _ in range(repeat):
                        self.samples.append((frame_path, label_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label_str = self.samples[idx]

        img = Image.open(frame_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label_id = self.label_encoder.encode(label_str)
        return img, label_id
