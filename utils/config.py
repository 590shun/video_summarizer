import os, sys
import shutil
import inspect
import logging
import datetime
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarizer.utils import parse_splits_filename
from summarizer.models.rand import RandomTrainer
from summarizer.models.logistic import LogisticRegressionTrainer
from summarizer.models.vasnet import VASNetTrainer
from summarizer.models.transformer import TransformerTrainer
from summarizer.models.dsn import DSNTrainer
from summarizer.models.sumgan import SumGANTrainer
from summarizer.models.sumgan_att import SumGANAttTrainer



class HParameters:
    def __init__(self):
        self.use_cuda = 'yes'
        self.cuda_device = 1
        self.weight_decay = 0.00001
        self.lr = 0.00005
        self.epochs = 50
        self.test_every_epochs = 50

        # dataset
        self.datasets = [
           'datasets/summarizer_dataset_summe_google_pool5.h5'
        ]

        # default split files to be trained/tested on
        self.splits_files = 'summe'

        # default model
        self.model_class = RandomTrainer

        # Dict containing extra parameters, possibly model-specific
        self.extra_params = None

        # summary length
        self.summary_proportion = 0.15

        # video segmentの選択
        self.selection_algorithm = 'knapsack'

        # logger default level is INFO
        self.log_level = logging.INFO

    def load_from_args(self, args):
        # any key from flags
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split(',')
                setattr(self, key, val)
        
        # pick model
        self.model_class = {
            'random': RandomTrainer,
            'logistic': LogisticRegressionTrainer,
            'vasnet': VASNetTrainer,
            'transformer': TransformerTrainer,
            'dsn': DSNTrainer,
            'sumgan': SumGANTrainer,
            'sumgan_att': SumGANAttTrainer,
            None: RandomTrainer
        }.get(args['model'], None)
        if self.model_class is None:
            raise KeyError(f"{args['model']} model is not unknown")


        # other dynamic properties
        self._init()

    def _init(self):
        # 実験名と出力先を指定
        log_dir = str(int(datetime.datetime.now().timestamp()))
        log_dir += '_' + self.model_class.__name__
        self.log_path = os.path.join('logs', log_dir)

        # Tensor Board
        self.writer = SummaryWriter(self.log_path)

        # cudaの扱いについて
        if self.use_cuda == 'default':
            self.use_cuda = torch.cuda.is_available()
        elif self.use_cuda == 'yes':
            self.use_cuda = True
        else:
            self.use_cuda = False

        # deviceの指定
        if self.use_cuda:
            torch.cuda.set_device(self.cuda_device)
        #     # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        # split fileの指定(defaultは'minimal')
        if self.splits_files == 'all':
            self.splits_files = [
                'splits/tvsum_splits.json',
                'splits/summe_splits.json']
        elif self.splits_files == 'tvsum':
            self.splits_files = ['splits/tvsum_splits.json']
        elif self.splits_files == 'summe':
            self.splits_files = ['splits/summe_splits.json']
        elif self.splits_files == 'dataset':
            self.splits_files = ['splits/dataset_splits.json']
        
        # file nameの管理リスト
        self.dataset_name_of_file = {}
        self.dataset_of_file = {}
        self.splits_of_file = {}

        for splits_file in self.splits_files:
            dataset_name, splits = parse_splits_filename(splits_file)
            self.dataset_name_of_file[splits_file] = dataset_name
            self.dataset_of_file[splits_file] = self.get_dataset_by_name(dataset_name).pop()
            self.splits_of_file[splits_file] = splits
        
        # destination for weights and predictions on dataset
        self.weights_path = {}
        self.pred_path = {}
        for splits_file in self.splits_files:
            weights_file = f'{os.path.basename(splits_file)}.pth'
            self.weights_path[splits_file] = os.path.join(self.log_path, weights_file)
            pred_file = f"{os.path.basename(splits_file)}_preds.h5"
            self.pred_path[splits_file] = os.path.join(self.log_path, pred_file)

        # logの保管先のディレクトリが存在しない場合にディレクトリを生成
        os.makedirs(self.log_path, exist_ok=True)

        # logger
        self.logger = logging.getLogger("summarizer")
        fmt = logging.Formatter("%(asctime)s::%(levelname)s: %(message)s", "%H:%M:%S")
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(self.log_path, "train.log"))
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.setLevel(getattr(logging, self.log_level.upper()))

        # modelをlog directoryの保存
        src = inspect.getfile(self.model_class)
        dst = os.path.join(self.log_path, os.path.basename(src))
        shutil.copyfile(src, dst)
    
    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None
    
    def __str__(self):
        # ハイパーパラメータを表示
        vars = ["use_cuda", "cuda_device", "log_level",
                "weight_decay", "lr", "epochs",
                "summary_proportion", "selection_algorithm",
                "log_path", "splits_files", "extra_params"]
        info_str = ""
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += "["+str(i)+"] "+var+": "+str(val)
            info_str += "\n" if i < len(vars)-1 else ""

        return info_str

    def get_full_hps_dict(self):
        """Returns the list of hyperparameters as a flat dict"""
        vars = ["weight_decay", "lr", "epochs"]

        hps = {}
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            hps[var] = val

        return hps

if __name__ == "__main__":
    # Check default values
    hps = HParameters()
    print(hps)
    # Check update with args works well
    args = {
        "root": "root_dir",
        "datasets": "set1,set2,set3",
        "splits": "split1, split2",
        "new_param_float": 1.23456
    }
    hps.load_from_args(args)
    print(hps)
