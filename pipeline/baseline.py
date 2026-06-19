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
from sagemaker.core.workflow.functions import Join




class Baseliner():

    def __init__(self, model_name_param, p_params, monitor_instance_type_param, sagemaker_session=None):
        self.model_name_param=model_name_param
        self.p_params=p_params   
        self.monitor_instance_type_param=monitor_instance_type_param
        self.sagemaker_session=sagemaker_session

    def get_make_baseline_sets_step(
            self, 
            role_param, 
            target_name_param, 
            prediction_name_param,
            build_role_arn,
            target_type_param,
            depends_on=[]
    ):
        # make create model step
        lambda_function = Lambda(
            function_name='MakeBaselineSets',
            execution_role_arn=build_role_arn,
            script='scripts/make_baseline_sets.py',  # path to your file
            handler='make_baseline_sets.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )

        create_model_step = LambdaStep(
            name='MakeBaselineSetsStep',
            lambda_func=lambda_function,
            inputs={
                'role': role_param,
                'baseline_file':self.p_params.baseline_file_param,
                'baseline_pred_file':self.p_params.baseline_pred_file_param,
                'dq_monitor_dir':self.p_params.dq_monitor_dir_param,
                'db_monitor_dir':self.p_params.db_monitor_dir_param,
                'mq_monitor_dir':self.p_params.mq_monitor_dir_param,
                'mb_monitor_dir':self.p_params.mb_monitor_dir_param,
                'me_monitor_dir':self.p_params.me_monitor_dir_param,
                'target_name':target_name_param,
                'prediction_name':prediction_name_param,
                'train_file':self.p_params.train_file_param,
                'train_X_file':self.p_params.train_X_file_param,
                'target_type':target_type_param
            },
            outputs=[
                LambdaOutput(output_name='result', output_type=LambdaOutputTypeEnum.String),
            ],
            depends_on=depends_on
        )
        return create_model_step

    def get_data_quality_baseline_step(self, role_param, depends_on=[]):

        dq_baseline_step = QualityCheckStep(
            name='DataQualityBaselineStep',
            quality_check_config=DataQualityCheckConfig(
                baseline_dataset=Join(on='/', values=[self.p_params.dq_monitor_dir_param, 'baseline.csv']),
                dataset_format=DatasetFormat.csv(header=True),
                output_s3_uri=Join(on='/', values=[self.p_params.dq_monitor_dir_param, 'info']),
            ),
            check_job_config=CheckJobConfig(
                role=role_param,
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


    def get_data_bias_baseline_step(self, target_param, role_param, depends_on=[]):
        db_baseline_step = ClarifyCheckStep(
            name='DataBiasBaselineStep',
            clarify_check_config=DataBiasCheckConfig(
                data_config=DataConfig(
                    s3_data_input_path=Join(on='/', values=[self.p_params.db_monitor_dir_param, 'baseline.csv']),
                    s3_output_path=Join(on='/', values=[self.p_params.db_monitor_dir_param, 'info']),
                    label=target_param,
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
                role=role_param,
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

    def get_model_quality_baseline_step(self, role_param, depends_on=[]):

        mq_baseline_step = QualityCheckStep(
            name='ModelQualityBaselineStep',
            
            quality_check_config=ModelQualityCheckConfig(
                problem_type="Regression",
                baseline_dataset=f'{self.p.dq_monitor_dir}/baseline.csv',
                dataset_format=DatasetFormat.csv(header=True),
                output_s3_uri=f'{self.p.dq_monitor_dir}/info'
            ),
            check_job_config=CheckJobConfig(
                role=role_param,
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


    def get_model_bias_baseline_step(self, role_param, depends_on=[]):

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
                role=role_param,
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

    def get_model_explainability_baseline_step(self, role_param, depends_on=[]):

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
                role=role_param,
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