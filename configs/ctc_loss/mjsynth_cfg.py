from easydict import EasyDict
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cfg = EasyDict()

cfg.root_dir = os.path.join(ROOT_DIR, 'data/mjsynth_mini/mnt/ramdisk/max/90kDICT32px')
cfg.cropped_data = True

cfg.alphabet = ["A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G", "g", "H", "h", "I", "i",
                "J", "j", "K", "k", "L", "l", "M", "m", "N", "n", "O", "o", "P", "p", "Q", "q", "R", "r",
                "S", "s", "T", "t", "U", "u", "V", "v", "W", "w", "X", "x", "Y", "y", "Z", "z",
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
cfg.blank_idx = 0

cfg.nrof_classes = len(cfg.alphabet) + 1

cfg.size = {
    'train': 1000000,
    'test': 100000
}

cfg.mean = 117.8204
cfg.std = 255
