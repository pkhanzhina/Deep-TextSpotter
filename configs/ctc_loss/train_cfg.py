import os
from datetime import datetime
from easydict import EasyDict
from configs.ctc_loss.mjsynth_cfg import cfg as dataset_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


cfg = EasyDict()

cfg.lr = 1e-5  # 1e-4
cfg.max_epoch = 10
cfg.device = 'gpu'
cfg.workers = 4

cfg.step_size = 10000
cfg.gamma = 0.98

cfg.evaluate_before_training = True
cfg.epoch_to_load = None

cfg.batch_size = 32  # 32
cfg.dataset = dataset_cfg

# checkpoint
cfg.checkpoints_dir = os.path.join(ROOT_DIR, 'data/ctc_loss', datetime.now().strftime('%Y-%m-%d_%H-%M'))
cfg.pretrained_dir = cfg.checkpoints_dir


neptune_cfg = EasyDict()
neptune_cfg.project_name = ''
neptune_cfg.api_token = ''
neptune_cfg.run_name = None

cfg.logger = neptune_cfg
