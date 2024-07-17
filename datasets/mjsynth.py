from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm


class MJSynth(Dataset):
    def __init__(self, cfg, data_type='train', transforms=None):
        self.cfg = cfg
        self.data_type = data_type
        self.transforms = transforms

        self.paths, self.labels = None, None
        self.word2idx = {self.cfg.alphabet[i]: i + 1 for i in range(len(self.cfg.alphabet))}
        self.idx2word = {i + 1: self.cfg.alphabet[i] for i in range(len(self.cfg.alphabet))}

        if self.cfg.cropped_data:  # для считывания части данных
            self.read_cropped_data()
        else:  # для считывания всего набора данных
            self.read_data()

        print(f"{data_type}: {len(self.paths)}, {len(self.labels)}")

    def read_data(self):
        annot_path = os.path.join(self.cfg.root_dir, f'annotation_{self.data_type}.txt')
        with open(annot_path, 'r') as f:
            annot = f.readlines()[:self.cfg.size[self.data_type]]
        paths, labels = [], []
        for sample in tqdm(annot, desc='Prepare data'):
            path = sample.split(' ')[0][2:]
            label = os.path.basename(path).split('_')[1]
            if not all(e in self.cfg.alphabet for e in label):
                continue
            paths.append(os.path.join(self.cfg.root_dir, path))
            labels.append(label)
        self.paths = paths
        self.labels = labels

    def read_cropped_data(self):
        annot_path = os.path.join(self.cfg.root_dir, f'annotation.txt')
        with open(annot_path, 'r') as f:
            annot = f.readlines()
        paths, labels = [], []
        for sample in tqdm(annot, desc='Prepare data'):
            path = sample.split(' ')[0][2:]
            if os.path.exists(os.path.join(self.cfg.root_dir, path)):
                label = os.path.basename(path).split('_')[1]
                if not all(e in self.cfg.alphabet for e in label):
                    continue
                paths.append(os.path.join(self.cfg.root_dir, path))
                labels.append(label)
                if len(labels) > 100:
                    break
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('L')
            label = list(map(lambda x: self.word2idx[x], self.labels[idx]))
            if self.transforms is not None:
                transformed_img = self.transforms(img)
            else:
                transformed_img = img
        except Exception as e:
            transformed_img, label = None, None
            print(f"{idx} ({self.paths[idx]}): {str(e)}")
        return transformed_img, label, self.labels[idx]
        # return {'data': transformed_img, 'label': label, 'word': self.labels[idx]}

    def seq_to_word(self, labels):
        return [''.join(list(map(lambda x: self.idx2word[x], l))) for l in labels]

    def get_mean(self, samples=10000):
        samples = min(samples, self.__len__())
        channel_sum, sum_pixel = 0, 0
        for ind in np.random.choice(np.arange(self.__len__()), samples, replace=False):
            img = np.asarray(Image.open(self.paths[ind]).convert('L'))
            sum_pixel += img.shape[0] * img.shape[1]
            channel_sum += np.sum(img)
        mean = channel_sum / sum_pixel
        print(F"Mean:\t {mean}")
        return mean

    def collate_fn(self, batch):
        batch = [b for b in batch if b[0] is not None]
        imgs, labels, words = list(zip(*batch))
        inputs_lengths = [img.shape[1] for img in imgs]
        max_len = np.max(inputs_lengths)
        inputs = [np.pad(img, ((0, 0), (0, max_len - inputs_lengths[i]))) for i, img in enumerate(imgs)]
        return torch.Tensor(np.asarray(inputs)).unsqueeze(1), labels, inputs_lengths, words


if __name__ == '__main__':
    from configs.ctc_loss.mjsynth_cfg import cfg
    from torchvision import transforms
    import datasets.augmentations as aug
    import numpy as np
    import matplotlib.pyplot as plt

    train_transforms = transforms.Compose([
        aug.VerticalResize(h=32),
        aug.RandomHorizontalCrop(scale=(0, 4)),
        aug.RandomHorizontalResize(scale=(0.8, 1.2)),
        aug.HorizontalResize(w=500),
        aug.GaussianNoise(mean=0, std=5)
    ])

    dataset = MJSynth(cfg, data_type='test', transforms=train_transforms)
    print(len(dataset))
    dataset.get_mean()

    # for item in np.random.randint(0, len(dataset), size=10):
    #     sample = dataset[item]
    #     img = np.asarray(sample[0])
    #     transformed_img = np.asarray(sample[1])
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img, cmap='gray')
    #     ax[1].imshow(transformed_img, cmap='gray')
    #     plt.title(sample[-1])
    #     plt.show()
    #     print(img.shape, transformed_img.shape)
    #     assert transformed_img.shape[0] == 32
