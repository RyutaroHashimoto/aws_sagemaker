import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.optim.adam import Adam


class TabNetBaseModel:
    """
    Note
    -----
    Reference
    -----
    https://github.com/dreamquark-ai/tabnet
    """

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def predict(self, X):
        return self.model.predict(X.values)

    def save_model(self, file_path):
        torch.save(self.model, file_path)

    def load_model(self, file_path):
        self.model = torch.load(file_path, map_location=torch.device(self.model.device))


class TabNet_PreTrainer(TabNetBaseModel):

    def fit(self, X_train):
        params = {
            'gamma': 1.3,
            'lambda_sparse': 0.001,
            'mask_type': 'entmax',
            'n_a': 8,
            'n_d': 8,
            'n_independent': 2,
            'n_shared': 2,
            'n_steps': 3,
            'optimizer_fn': Adam,
            'optimizer_params': {'lr': 0.02, 'weight_decay': 1e-05},
            'scheduler_fn': torch.optim.lr_scheduler.OneCycleLR,
            'scheduler_params': {'epochs': 200, 'max_lr': 0.05, 'steps_per_epoch': 3},
            'seed': 123,
            'verbose': 10,
            'device_name': 'auto',
        }

        params.update(self.params)
        self.params = params

        params_tabnet = {
            'X_train': X_train.values,
            'eval_set': [X_train.values],
            'max_epochs': params.pop('max_epochs', 200),
            'patience': params.pop('patience', 20),
            'batch_size': params.pop('batch_size', 256),
            'virtual_batch_size': params.pop('virtual_batch_size', 128),
            'num_workers': 1,
            'drop_last': True,
        }

        self.model = TabNetPretrainer(**self.params)
        self.model.fit(**params_tabnet)


class TabNet_Regressor(TabNetBaseModel):

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        params = {
            'n_d': 8,
            'n_a': 8,
            'n_steps': 1,
            'gamma': 1.3,
            'n_independent': 2,
            'n_shared': 2,
            'seed': 1234,
            'lambda_sparse': 0.001,
            'optimizer_fn': Adam,
            'optimizer_params': {'lr': 0.02, 'weight_decay': 1e-05},
            'mask_type': 'entmax',
            'scheduler_params': {'max_lr': 0.05, 'steps_per_epoch': 3, 'epochs': 200},
            'scheduler_fn': torch.optim.lr_scheduler.OneCycleLR,
            'verbose': 10,
            'device_name': 'auto',
        }

        params.update(self.params)
        self.params = params

        # if validation data is given
        validation = X_valid is not None
        if validation:
            params.update({'eval_set': [(X_valid.values, y_valid.values.reshape(-1, 1))]})

        params_tabnet = {
            'X_train': X_train.values,
            'y_train': y_train.values.reshape(-1, 1),
            'eval_name': params.pop('eval_name', None),
            'eval_metric': params.pop('eval_metric', None),
            'max_epochs': params.pop('max_epochs', 200),
            'patience': params.pop('patience', 30),
            'batch_size': params.pop('batch_size', 128),
            'virtual_batch_size': params.pop('virtual_batch_size', 64),
            'from_unsupervised': params.pop('from_unsupervised', None),
            'eval_set': params.pop('eval_set', []),
            'num_workers': 1,
            'drop_last': False,
        }

        self.model = TabNetRegressor(**self.params)
        self.model.fit(**params_tabnet)
