import pandas as pd
from sagemaker.core.model_monitor.model_monitoring import DefaultModelMonitor, ModelQualityMonitor
from sagemaker.core.model_monitor.clarify_model_monitoring import ModelBiasMonitor, ModelExplainabilityMonitor
from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.model_monitor.dataset_format import DatasetFormat
from sagemaker.core.network import NetworkConfig
from sagemaker.core.clarify import DataConfig, BiasConfig, ModelConfig, ModelPredictedLabelConfig, SHAPConfig
from sagemaker.core.shapes.shapes import ModelDashboardMonitoringSchedule, MonitoringScheduleConfig, BatchTransformInput, MonitoringDatasetFormat, MonitoringCsvDatasetFormat, MonitoringResources, MonitoringClusterConfig, MonitoringAppSpecification, MonitoringJobDefinition, MonitoringInput, MonitoringOutputConfig, MonitoringS3Output, MonitoringOutput, MonitoringStoppingCondition, ScheduleConfig
from sagemaker.core.image_uris import retrieve as retrieve_image
from sagemaker.mlops.workflow.steps import ProcessingStep
from sagemaker.mlops.workflow.quality_check_step import ModelQualityCheckConfig, DataQualityCheckConfig
from sagemaker.mlops.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.mlops.workflow.quality_check_step import QualityCheckStep, DataQualityCheckConfig
from sagemaker.mlops.workflow.check_job_config import CheckJobConfig
from sagemaker.mlops.workflow.clarify_check_step import ClarifyCheckStep, ModelBiasCheckConfig, ModelExplainabilityCheckConfig, DataBiasCheckConfig
from sagemaker.mlops.workflow.lambda_step import LambdaOutputTypeEnum, LambdaStep, LambdaOutput
from sagemaker.core.lambda_helper import Lambda




class Baseliner():

    def __init__(self, model_name, paths, monitor_instance_type, sagemaker_session=None):
        self.model_name=model_name
        self.p= paths   
        self.monitor_instance_type=monitor_instance_type
        self.sagemaker_session=sagemaker_session

    def make_baseline_sets(self, target_name, prediction_name, target_type=float):

        baseline=pd.read_csv(self.p.baseline_file, header=0)
        baseline_pred=pd.read_csv(self.p.baseline_pred_file, header=None)
        baseline_pred.columns=[prediction_name]
        baseline_full = pd.concat([baseline_pred, baseline], axis=1)
        baseline_full[target_name] = baseline_full[target_name].astype(target_type)
        baseline_full[prediction_name] = baseline_full[prediction_name].astype(target_type)

        # Data Quality → input features only
        baseline_full.drop(columns=[target_name, prediction_name]).to_csv(f'{self.p.dq_monitor_dir}/baseline.csv', index=False, header=True)

        # Data Bias → input features + target
        baseline_full.drop(columns=[prediction_name]).to_csv(f'{self.p.dq_monitor_dir}/baseline.csv', index=False, header=True)

        # Model Quality → predictions + ground truth labels
        baseline_full[[target_name, prediction_name]].to_csv(f'{self.p.mq_monitor_dir}/baseline.csv', index=False, header=True)

        # Model Bias → features + predictions + labels
        baseline_full.to_csv(f'{self.p.mb_monitor_dir}/baseline.csv', index=False, header=True)

        # Model Explainability → input features + predictions (uses SHAP values)
        baseline_full.drop(columns=[target_name]).to_csv(f'{self.p.me_monitor_dir}/baseline.csv', index=False, header=True)

        train=pd.read_csv(self.p.train_file, header=None)
        train_X = train.iloc[:, 1:]
        train_X.to_csv(self.p.train_X_file, index=False, header=False)

    def get_make_baseline_sets_step(
            self, 
            role_param, 
            target_name, 
            prediction_name, 
            depends_on=[]
    ):
        # make create model step
        lambda_function = Lambda(
            function_name='MakeBaselineSets',
            execution_role_arn=role_param.default_value,
            script='scripts/make_baseline_sets.py',  # path to your file
            handler='make_baseline_sets.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )

        create_model_step = LambdaStep(
            name='MakeBaselineSetsStep',
            lambda_func=lambda_function,
            inputs={
                'baseline_file':self.p.baseline_file,
                'baseline_pred_file':self.p.baseline_pred_file,
                'dq_monitor_dir':self.p.dq_monitor_dir,
                'db_monitor_dir':self.p.db_monitor_dir,
                'mq_monitor_dir':self.p.mq_monitor_dir,
                'mb_monitor_dir':self.p.mb_monitor_dir,
                'me_monitor_dir':self.p.me_monitor_dir,
                'target_name':target_name,
                'prediction_name':prediction_name,
                'train_file':self.p.train_file,
                'train_X_file':self.p.train_X_file,
                'target_type':self.p.target_type
            },
            outputs=[
                LambdaOutput(output_name='result', output_type=LambdaOutputTypeEnum.String),
            ],
            depends_on=depends_on
        )
        return create_model_step

    def get_data_quality_baseline_step(self, action, role, depends_on=[]):

        dq_baseline_step = QualityCheckStep(
            name='DataQualityBaselineStep',
            quality_check_config=DataQualityCheckConfig(
                baseline_dataset=f'{self.p.dq_monitor_dir}/baseline.csv',
                dataset_format=DatasetFormat.csv(header=True),
                output_s3_uri=f'{self.p.dq_monitor_dir}/info'
            ),
            check_job_config=CheckJobConfig(
                role=role,
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=20,
                max_runtime_in_seconds=1800,
                sagemaker_session=self.sagemaker_session
            ),
            skip_check=True,
            register_new_baseline=True,
            depends_on=depends_on
        )
       
        return dq_baseline_step


    def get_data_bias_baseline_step(self, role, depends_on=[]):
        db_baseline_step = ClarifyCheckStep(
            name='DataBiasBaselineStep',
            clarify_check_config=DataBiasCheckConfig(
                data_config=DataConfig(
                    s3_data_input_path=f'{self.p.db_monitor_dir}/baseline.csv',
                    s3_output_path=f'{self.p.db_monitor_dir}/info',
                    label='rings',
                    dataset_type='text/csv'
                ),
                data_bias_config=BiasConfig(
                    label_values_or_threshold=[7],
                    facet_name='sex_F',
                    facet_values_or_threshold=[1]
                ),
                methods='all'
            ),
            check_job_config=CheckJobConfig(
                role=role,
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=20,
                max_runtime_in_seconds=1800,
                sagemaker_session=self.sagemaker_session
            ),
            skip_check=True,
            register_new_baseline=True,
            depends_on=depends_on
        )

        return db_baseline_step

    def get_model_quality_baseline_step(self, role, depends_on=[]):

        mq_baseline_step = QualityCheckStep(
            name='ModelQualityBaselineStep',
            
            quality_check_config=ModelQualityCheckConfig(
                problem_type="Regression",
                baseline_dataset=f'{self.p.dq_monitor_dir}/baseline.csv',
                dataset_format=DatasetFormat.csv(header=True),
                output_s3_uri=f'{self.p.dq_monitor_dir}/info'
            ),
            check_job_config=CheckJobConfig(
                role=role,
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=20,
                max_runtime_in_seconds=1800,
                sagemaker_session=self.sagemaker_session
            ),
            skip_check=True,           # True for baseline creation
            register_new_baseline=True, # register the new baseline
            depends_on=depends_on
        )

        return mq_baseline_step


    def get_model_bias_baseline_step(self, role, depends_on=[]):

        mb_baseline_step = ClarifyCheckStep(
            name='ModelBiasBaselineStep',
            clarify_check_config=ModelBiasCheckConfig(
                data_config=DataConfig(
                    s3_data_input_path=f'{self.p.mb_monitor_dir}/baseline.csv',
                    s3_output_path=f'{self.p.mb_monitor_dir}/info',
                    label='rings',
                    predicted_label='rings_prediction',
                    dataset_type='text/csv',

                ),
                model_predicted_label_config=ModelPredictedLabelConfig(
                    label='rings', 
                    probability = None, 
                    probability_threshold = None, 
                    label_headers = None
                ),
                data_bias_config=BiasConfig(
                    label_values_or_threshold=[7],
                    facet_name='sex_F',
                    facet_values_or_threshold=[1]
                ),
                model_config=ModelConfig(
                    model_name=self.p.model_name,
                    instance_type=self.p.monitor_instance_type,
                    instance_count=1,
                    accept_type='text/csv',
                    content_type='text/csv'
                ),
                methods='all'
            ),
            check_job_config=CheckJobConfig(
                role=role,
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=20,
                max_runtime_in_seconds=1800,
                sagemaker_session=self.sagemaker_session
            ),
            skip_check=True,             # no baseline to compare against yet
            register_new_baseline=True,  # save this as the new baseline
            depends_on=depends_on
        )

        return mb_baseline_step

    def get_model_explainability_baseline_step(self, role, depends_on=[]):

        X_train = pd.read_csv(self.p.train_X_file, header=None)

        me_baseline_step = ClarifyCheckStep(
            name='ModelExplainabilityBaselineStep',
            clarify_check_config=ModelExplainabilityCheckConfig(
                data_config=DataConfig(
                    s3_data_input_path=f'{self.p.me_monitor_dir}/baseline.csv',
                    s3_output_path=f'{self.p.me_monitor_dir}/info',
                    dataset_type='text/csv'
                ),
                model_config=ModelConfig(
                    model_name=self.p.model_name,
                    instance_type=self.p.monitor_instance_type,
                    instance_count=1,
                    accept_type='text/csv',
                    content_type='text/csv'
                ),
                explainability_config=SHAPConfig(
                    baseline=[X_train.mean().tolist()],
                    num_samples=100,
                    agg_method='mean_abs'
                )
            ),
            check_job_config=CheckJobConfig(
                role=role,
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=20,
                max_runtime_in_seconds=1800,
                sagemaker_session=self.sagemaker_session
            ),
            skip_check=True,
            register_new_baseline=True,
            depends_on=depends_on
        )

        return me_baseline_step
    

















    # def get_data_quality_step(self, action, role, depends_on=[]):

    #     if action == 'deploy':
    #         name='DataQualityBaselineStep'
    #         baseline_dataset=f'{self.p.dq_monitor_dir}/baseline.csv',
    #         output_s3_uri=f'{self.p.dq_monitor_dir}/info'
    #         skip_check=True
    #         register_new_baseline=True
    #         supplied_baseline_statistics=None,
    #         supplied_baseline_constraints=None,
    #     else:
    #         name='DataQualityMonitorStep'
    #         baseline_dataset=f'{self.p.dq_monitor_dir}/baseline.csv',
    #         output_s3_uri=f'{self.p.dq_monitor_dir}/check_output'
    #         skip_check=False
    #         register_new_baseline=False
    #         supplied_baseline_statistics=f'{self.p.dq_monitor_dir}/info/constraints.json',
    #         supplied_baseline_constraints=f'{self.p.dq_monitor_dir}/info/statistics.json',

    #     dq_baseline_step = QualityCheckStep(
    #         name=name,
    #         quality_check_config=DataQualityCheckConfig(
    #             baseline_dataset=baseline_dataset,
    #             dataset_format=DatasetFormat.csv(header=True),
    #             output_s3_uri=output_s3_uri
    #         ),
    #         check_job_config=CheckJobConfig(
    #             role=role,
    #             instance_count=1,
    #             instance_type=self.monitor_instance_type,
    #             volume_size_in_gb=20,
    #             max_runtime_in_seconds=1800,
    #             sagemaker_session=self.sagemaker_session
    #         ),
    #         skip_check=skip_check,
    #         register_new_baseline=register_new_baseline,
    #         supplied_baseline_statistics=supplied_baseline_statistics,
    #         supplied_baseline_constraints=supplied_baseline_constraints,
    #         depends_on=depends_on
    #     )
       
    #     return dq_baseline_step