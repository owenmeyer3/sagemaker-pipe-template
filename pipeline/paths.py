from sagemaker.core.workflow.functions import Join

class Paths():
    def __init__(self, bucket_name, project_name, model_prefix):
        self.project_dir=f's3://{bucket_name}/{project_name}'
        self.model_instance_dir=f'{self.project_dir}/models/{model_prefix}'
        self.data_dir=f'{self.model_instance_dir}/data'
        self.model_dir=f'{self.model_instance_dir}/model'
        self.temp_data_dir=f'{self.data_dir}/temp'

        # model inputs
        self.train_dir=f'{self.data_dir}/input/train'
        self.validation_dir=f'{self.data_dir}/input/validation'
        self.test_dir=f'{self.data_dir}/input/test'
        self.train_file=f'{self.train_dir}/train.csv'
        self.validation_file=f'{self.validation_dir}/validation.csv'
        self.test_file=f'{self.test_dir}/test.csv'
        self.test_X_file=f'{self.test_dir}/test_X.csv'
        self.test_y_file=f'{self.test_dir}/test_y.csv'

        # baselining
        self.baseline_dir=  f'{self.data_dir}/baseline'
        self.baseline_file=  f'{self.baseline_dir}/baseline.csv'
        self.baseline_X_file=f'{self.baseline_dir}/baseline_X.csv'
        self.baseline_model_out_file= f'{self.temp_data_dir}/baseline_X.csv.out'
        self.baseline_y_file=f'{self.baseline_dir}/baseline_y.csv'
        self.temp_data_dir=f'{self.data_dir}/temp'


        # Batch
        self.batch_in_dir=   f'{self.data_dir}/batch-input'
        self.batch_out_dir=  f'{self.data_dir}/batch-output'

        # Deploy
        self.data_capture_dir=  f'{self.data_dir}/capture'
        self.ground_truth_dir=  f'{self.data_dir}/ground-truth'

        # Monitors
        self.monitors_dir=  f'{self.data_dir}/monitors'
        self.dq_monitor_dir=f'{self.monitors_dir}/data-quality'
        self.mq_monitor_dir=f'{self.monitors_dir}/model-quality'
        self.mb_monitor_dir=f'{self.monitors_dir}/model-bias'
        self.me_monitor_dir=f'{self.monitors_dir}/model-explainability'

        

class PathParams():
    def __init__(self, training_bucket_param, training_dir_param, pipeline_bucket_param, name):
        #### TRAIN ####
        self.training_dir_param = Join(on='/', values=['s3:/', training_bucket_param, training_dir_param])
        self.data_dir=Join(on='/', values=[self.training_dir_param, 'data'])
        self.model_dir=Join(on='/', values=[self.training_dir_param, 'model'])

        # model inputs
        self.train_dir=       Join(on='/', values=[self.data_dir, 'input/train'])
        self.validation_dir=  Join(on='/', values=[self.data_dir, 'input/validation'])
        self.test_dir=        Join(on='/', values=[self.data_dir, 'input/test'])
        self.train_file=      Join(on='/', values=[self.train_dir, 'train.csv'])
        self.validation_file= Join(on='/', values=[self.validation_dir, 'validation.csv'])
        self.test_file=       Join(on='/', values=[self.test_dir, 'modeltest.csv'])
        self.test_X_file=     Join(on='/', values=[self.test_dir, 'test_X.csv'])
        self.test_y_file=     Join(on='/', values=[self.test_dir, 'test_y.csv'])

        # baselining
        self.baseline_dir=            Join(on='/', values=[self.data_dir, 'baseline'])
        self.baseline_file=           Join(on='/', values=[self.baseline_dir, 'baseline.csv'])
        self.baseline_X_file=         Join(on='/', values=[self.baseline_dir, 'baseline_X.csv'])
        self.baseline_model_out_file= Join(on='/', values=[self.temp_data_dir, 'baseline_X.csv.out'])
        self.baseline_y_file=         Join(on='/', values=[self.baseline_dir, 'baseline_y.csv'])

        ### Pipeline ###
        self.pipeline_dir_param = Join(on='/', values=['s3:/', pipeline_bucket_param, 'pipelines', name])

        self.temp_data_dir_param= Join(on='/', values=[self.pipeline_dir_param, 'temp'])

        # Batch
        self.batch_in_dir_param=  Join(on='/', values=[self.pipeline_dir_param, 'batch-input'])
        self.batch_out_dir_param= Join(on='/', values=[self.pipeline_dir_param, 'batch-output'])

        # Deploy
        self.data_capture_dir_param= Join(on='/', values=[self.pipeline_dir_param, 'capture'])
        self.ground_truth_dir_param= Join(on='/', values=[self.pipeline_dir_param, 'ground-truth'])

        # Monitors
        self.monitors_dir_param=   Join(on='/', values=[self.pipeline_dir_param, 'monitors'])
        self.dq_monitor_dir_param= Join(on='/', values=[self.monitors_dir_param, 'data-quality'])
        self.mq_monitor_dir_param= Join(on='/', values=[self.monitors_dir_param, 'model'])
        self.mb_monitor_dir_param= Join(on='/', values=[self.monitors_dir_param, 'model-bias'])
        self.me_monitor_dir_param= Join(on='/', values=[self.monitors_dir_param, 'model-explainability'])


        