class LabelEncoder:
    def __init__(self, class_names):
        self.label2id = {label: idx for idx, label in enumerate(class_names)}
        self.id2label = {idx: label for idx, label in enumerate(class_names)}

    def encode(self, label):
        if label not in self.label2id:
            raise ValueError(f"Unknown label: {label}")
        return self.label2id[label]

    def decode(self, idx):
        if idx not in self.id2label:
            raise ValueError(f"Unknown label id: {idx}")
        return self.id2label[idx]

    @property
    def classes_(self):
        return list(self.label2id.keys())
