
class Paths():
    def __init__(self, bucket_name, project_name, model_prefix):
        self.project_dir=f's3://{bucket_name}/{project_name}/models/{model_prefix}'
        self.data_dir=f'{self.project_dir}/data'
        self.temp_data_dir=f'{self.project_dir}/data/temp'
        self.model_dir=f'{self.project_dir}/model'

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