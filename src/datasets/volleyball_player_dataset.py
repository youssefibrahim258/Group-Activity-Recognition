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

        with open(pickle,"rb") as f :
            self.videos_annot=pickle.load(f)

        self.samples=[]

        for video_dir in video_list :
            if video_dir not in self.videos_annot:
                continue

            clips=self.videos_annot[video_dir]

            for clip_dir,clip_data in clips.items():
                frames=
