import datetime
import json

import pandas as pd
import sagemaker


class RunSageMaker:
    """
    インスタンスを立ち上げ学習を実行する。
    学習結果を登録しているmetricsを含んだpandas DataFrameで返す。
    """

    def __init__(self,
                 image_uri,
                 role,
                 params,
                 metric_definitions,
                 tag_config,
                 instance_config=None):
        self.estimator = None
        self.image_uri = image_uri
        self.role = role
        self.metric_definitions = metric_definitions
        self.tag_config = tag_config

        if params is None:
            self.params = {}
        else:
            self.params = params
        if instance_config is None:
            self.instance_config = {}
        else:
            self.instance_config = instance_config

    def fit(self, input, output_path, job_name=None, tag_config=None):
        if job_name is None:
            self.job_name = 'sagemaker-' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        else:
            self.job_name = job_name

        if tag_config is not None:
            self.tag_config = {**self.tag_config, **tag_config}

        self.params['input'] = json.dumps(input)

        # initialize instance
        self.estimator_params = {
            'image_uri': self.image_uri,
            'role': self.role,
            'output_path': output_path,
            'enable_sagemaker_metrics': True,
            'hyperparameters': self.params,
            'tags': self.tag_config,
            'metric_definitions': self.metric_definitions,
        }
        self.estimator_params = {**self.estimator_params, **self.instance_config, }

        self.estimator = sagemaker.estimator.Estimator(**self.estimator_params)

        self.estimator.fit(input['base_dir'], job_name=self.job_name)

    def deploy(self,
               initial_instance_count=1,
               instance_type='local',
               endpoint_name=None,
               serializers=sagemaker.serializers.CSVSerializer(),
               deserializer=sagemaker.deserializers.CSVDeserializer()):
        """
        推論用エンドポイントを作成する。
        """
        self.predictor = self.estimator.deploy(initial_instance_count, instance_type, endpoint_name=endpoint_name)
        self.predictor.serializer = serializers
        self.predictor.deserializer = deserializer

    def predict(self, x):
        """
        推論用エンドポイントを使用して予測する。
        """
        if type(self.predictor.serializer) == sagemaker.serializers.CSVSerializer:
            if type(x) is pd.core.frame.DataFrame:
                x = x.values
            res = self.predictor.predict(x)
            res = [float(s[0]) for s in res]
        else:
            res = self.predictor.predict(x)
        return res

    def delete_endpoint(self):
        """
        エンドポイントを削除
        """
        self.predictor.delete_endpoint()
