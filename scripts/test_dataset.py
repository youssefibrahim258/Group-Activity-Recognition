import matplotlib.pyplot as plt
from torchvision import transforms

from src.datasets.volleyball_clip_dataset import VolleyballClipDataset
from src.utils.label_encoder import LabelEncoder


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    class_names = [
        "r_set", "r_spike", "r-pass", "r_winpoint",
        "l_winpoint", "l-pass", "l-spike", "l_set"
    ]

    encoder = LabelEncoder(class_names=class_names)

    dataset = VolleyballClipDataset(
        videos_dir=r"data_set\videos_sample",
        video_ids=["7"],
        label_encoder=encoder,
        transform=transform
    )

    print("Dataset size:", len(dataset))

    image1, label_id1,img_path = dataset[0]
    label_name = encoder.decode(label_id1)

    print("Label id:", label_id1)
    print("Label name:", label_name)
    print(img_path)

    # img = image.permute(1, 2, 0)
    # plt.imshow(img)
    # plt.title(f"Class: {label_name}")
    # plt.axis("off")
    # plt.show()


if __name__ == "__main__":
    main()
