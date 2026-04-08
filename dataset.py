from torch.utils.data import Dataset
import os

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # ✅ Correct loading for .npy
        image = np.load(img_path)
        mask = np.load(mask_path)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Fix shape
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask