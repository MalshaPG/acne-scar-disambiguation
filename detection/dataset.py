import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import os


class ACNE04Dataset(Dataset):
    """
    Loads ACNE04 images together with their bounding-box annotations
    (from Detection/VOC2007/Annotations) and severity label + lesion count
    (from the Classification split files: filename, severity[0-3], lesion_count).

    Verified against source: severity/lesion_count columns confirmed to match
    filename prefix (levleN_) and XML object count respectively (see
    notebooks/01_acne04_exploration.ipynb).
    """

    def __init__(self, root, split_file, transform=None):
        self.image_dir = os.path.join(root, 'classification/Classification/JPEGImages')
        self.ann_dir = os.path.join(root, 'detection/Detection/VOC2007/Annotations')
        self.transform = transform

        split_path = os.path.join(root, 'classification/Classification', split_file)
        self.entries = []
        with open(split_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    filename, severity, lesion_count = parts[0], int(parts[1]), int(parts[2])
                    self.entries.append((filename, severity, lesion_count))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filename, severity, lesion_count = self.entries[idx]

        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')

        xml_path = os.path.join(self.ann_dir, filename.replace('.jpg', '.xml'))
        boxes = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root_el = tree.getroot()
            for obj in root_el.findall('object'):
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'boxes': boxes,
            'severity': severity,
            'lesion_count': lesion_count,
            'filename': filename
        }