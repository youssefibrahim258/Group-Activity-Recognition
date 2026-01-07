from torch.utils.data import Dataset
from PIL import Image
import torch
import pickle
import os
class VolleyballB1Dataset(Dataset):
    def __init__(self, pickle_file, videos_root, video_list, encoder, transform=None):
        self.videos_root = videos_root
        self.transform = transform
        self.encoder = encoder  

        with open(pickle_file, 'rb') as f:
            self.videos_annot = pickle.load(f)

        self.samples = []
        for video_dir in video_list:
            if video_dir not in self.videos_annot:
                continue
            clips = self.videos_annot[video_dir]
            for clip_dir, clip_data in clips.items():
                frames = sorted(clip_data['frame_boxes_dct'].keys())
                if len(frames) == 0:
                    continue
                middle_frame = frames[len(frames)//2]
                img_path = os.path.join(videos_root, video_dir, clip_dir, f"{middle_frame}.jpg")
                label_str = clip_data['category']
                label_int = self.encoder.encode(label_str)
                self.samples.append((img_path, label_int))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
