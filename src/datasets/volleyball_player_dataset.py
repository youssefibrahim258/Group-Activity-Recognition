from torch.utils.data import Dataset
from PIL import Image
import pickle
import os


class VolleyballB3Dataset(Dataset):

    def __init__(self,pickle_file,videos_root,video_list,encoder,transform=None,multiple_frames=False):

        self.videos_root = videos_root
        self.transform = transform
        self.encoder = encoder
        # standing_count = 0
        # MAX_STANDING = 6000
        # STANDING_CLASS = 8

        self.samples = []   # (img_path, box_coords, label_int)
        self.labels = []    # labels only (for class weights)

        # Load annotations
        with open(pickle_file, "rb") as f:
            self.videos_annot = pickle.load(f)

        for video_dir in video_list:
            video_dir = str(video_dir)
            if video_dir not in self.videos_annot:
                print(f"Video {video_dir} not found in annotations")
                continue

            clips = self.videos_annot[video_dir]

            for clip_dir, clip_data in clips.items():
                frame_ids = sorted(clip_data["frame_boxes_dct"].keys())
                if len(frame_ids) == 0:
                    continue

                if multiple_frames:
                    mid_idx = len(frame_ids) // 2
                    start = max(0, mid_idx - 5)
                    end = min(len(frame_ids), mid_idx + 5)
                    selected_frame_ids = frame_ids[start:end]
                else:
                    selected_frame_ids = [frame_ids[len(frame_ids) // 2]]

                for frame_id in selected_frame_ids:
                    boxes = clip_data["frame_boxes_dct"][frame_id]
                    img_path = os.path.join(
                        videos_root, video_dir, clip_dir, f"{frame_id}.jpg"
                    )

                    for box in boxes:
                        label_int = self.encoder.encode(box.category)

                        # if label_int == STANDING_CLASS:
                        #     if standing_count >= MAX_STANDING:
                        #         continue
                        #     standing_count += 1

                        self.samples.append((img_path, box.box, label_int))
                        self.labels.append(label_int)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, box_coords, label_int = self.samples[idx]
        x1, y1, x2, y2 = box_coords
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, x1 - 0.10 * w)
        y1 = max(0, y1 - 0.10 * h)
        x2 = x2 + 0.10 * w
        y2 = y2 + 0.10 * h

        img = Image.open(img_path).convert("RGB")
        img_crop = img.crop((x1, y1, x2, y2))

        if self.transform:
            img_crop = self.transform(img_crop)

        return img_crop, label_int
