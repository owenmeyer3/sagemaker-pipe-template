import pandas as pd
from sagemaker.core.model_monitor.model_monitoring import DefaultModelMonitor, ModelQualityMonitor
from sagemaker.core.model_monitor.clarify_model_monitoring import ModelBiasMonitor, ModelExplainabilityMonitor
from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.model_monitor.dataset_format import DatasetFormat
from sagemaker.core.network import NetworkConfig
from sagemaker.core.clarify import DataConfig, BiasConfig, ModelConfig, ModelPredictedLabelConfig, SHAPConfig
from sagemaker.core.shapes.shapes import ModelDashboardMonitoringSchedule, MonitoringScheduleConfig, BatchTransformInput, EndpointInput, MonitoringDatasetFormat, MonitoringCsvDatasetFormat, MonitoringResources, MonitoringClusterConfig, MonitoringAppSpecification, MonitoringJobDefinition, MonitoringInput, MonitoringOutputConfig, MonitoringS3Output, MonitoringOutput, MonitoringStoppingCondition, ScheduleConfig
from sagemaker.core.image_uris import retrieve as retrieve_image
from sagemaker.mlops.workflow.steps import ProcessingStep
from sagemaker.mlops.workflow.quality_check_step import ModelQualityCheckConfig, DataQualityCheckConfig
from sagemaker.mlops.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.mlops.workflow.quality_check_step import QualityCheckStep, DataQualityCheckConfig
from sagemaker.mlops.workflow.check_job_config import CheckJobConfig

def get_monitor_schedule_config(deploy_type, role, monitor_type, schedule_expression, data_cature_dir, monitor_dir, constraints_path, statistics_path, endpoint_name=None, instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800):

    if deploy_type == 'realtime':
        monitoring_inputs=[{
            "EndpointInput": {
                "EndpointName": "abalone-endpoint",
                "LocalPath": "/opt/ml/processing/input/endpoint"
            }
        }]
    else:
        monitoring_inputs=[{
            "BatchTransformInput": {
                "DataCapturedDestinationS3Uri": f"{data_cature_dir}/",
                "LocalPath": "/opt/ml/processing/input",
                "DatasetFormat": {
                    "Csv": {"Header": False}
                }
            }
        }]

    return {
        "ScheduleConfig": {
            "ScheduleExpression": schedule_expression# "cron(0 * ? * * *)"
        },
        "MonitoringJobDefinition": {
            "BaselineConfig": {
                "ConstraintsResource": {"S3Uri": constraints_path},
                "StatisticsResource": {"S3Uri": statistics_path}
            },
            "MonitoringInputs": monitoring_inputs,
            "MonitoringOutputConfig": {
                "MonitoringOutputs": [{
                    "S3Output": {
                        "S3Uri": f'{monitor_dir}/reports',
                        "LocalPath": "/opt/ml/processing/output"
                    }
                }]
            },
            "MonitoringResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": instance_type,
                    "VolumeSizeInGB": volume_size_in_gb
                }
            },
            "MonitoringAppSpecification": {
                "ImageUri": "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer"
            },
            "RoleArn": role,
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_in_seconds}
        },
        "MonitoringType": monitor_type #"DataQuality"
    }

def create_scheduled_data_quality_monitor(boto_session, role, name, deploy_type, schedule_expression, constraints_path, statistics_path, monitor_dir, endpoint_name=None, data_cature_dir=None, instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800):

    sm_client = boto_session.client('sagemaker', region_name='us-east-1')

    if deploy_type == 'realtime':
        monitoring_inputs=[{
            "EndpointInput": {
                "EndpointName": endpoint_name,
                "LocalPath": "/opt/ml/processing/input/endpoint"
            }
        }]
    else:
        monitoring_inputs=[{
            "BatchTransformInput": {
                "DataCapturedDestinationS3Uri": f"{data_cature_dir}/",
                "LocalPath": "/opt/ml/processing/input",
                "DatasetFormat": {
                    "Csv": {"Header": False}
                }
            }
        }]

    sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
            MonitoringScheduleConfig={
            "ScheduleConfig": {
                "ScheduleExpression": schedule_expression# "cron(0 * ? * * *)"
            },
            "MonitoringJobDefinition": {
                "BaselineConfig": {
                    "ConstraintsResource": {"S3Uri": constraints_path},
                    "StatisticsResource": {"S3Uri": statistics_path}
                },
                "MonitoringInputs": monitoring_inputs,
                "MonitoringOutputConfig": {
                    "MonitoringOutputs": [{
                        "S3Output": {
                            "S3Uri": f'{monitor_dir}/reports',
                            "LocalPath": "/opt/ml/processing/output"
                        }
                    }]
                },
                "MonitoringResources": {
                    "ClusterConfig": {
                        "InstanceCount": 1,
                        "InstanceType": instance_type,
                        "VolumeSizeInGB": volume_size_in_gb
                    }
                },
                "MonitoringAppSpecification": {
                    "ImageUri": "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer"
                },
                "RoleArn": role,
                "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_in_seconds}
            },
            "MonitoringType": "DataQuality"
        }
    )

def create_scheduled_model_quality_monitor(boto_session, role, monitor_type, name, deploy_type, schedule_expression, constraints_path, statistics_path, monitor_dir, endpoint_name=None, data_cature_dir=None, instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800):

    sm_client = boto_session.client('sagemaker', region_name='us-east-1')

    sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig=get_monitor_schedule_config(
            deploy_type, 
            role, 
            monitor_type, 
            schedule_expression, 
            data_cature_dir, monitor_dir, 
            constraints_path, 
            statistics_path,
            endpoint_name=endpoint_name,
            instance_type=instance_type, 
            volume_size_in_gb=volume_size_in_gb, 
            max_runtime_in_seconds=max_runtime_in_seconds
        )
    )

# def get_monitor_batch_transform_step(self, sagemaker_session, role, create_model_step, scope, writes={}, depends_on=[]):
# {
#     'ModelName': 'sagemaker-xgboost-2026-05-21-17-13-20-923',
#     'TransformInput': {'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
#         'S3Uri': 's3://omm-test-bucket/models/abalone/data/input/test/test_X.csv'}},
#     'ContentType': 'text/csv',
#     'SplitType': 'Line'},
#     'TransformOutput': {'S3OutputPath': 's3://omm-test-bucket/models/test/data/transformations/out',
#     'Accept': 'text/csv',
#     'AssembleWith': 'Line'},
#     'TransformResources': {'InstanceType': 'ml.m5.large', 'InstanceCount': 1}
# }
# def get_monitor_batch_transform_step(self, sagemaker_session, role, create_model_step, scope, writes={}, depends_on=[]):
# {
#     'ModelName': 'sagemaker-xgboost-2026-05-21-17-13-20-923',
#     'TransformInput': {'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
#         'S3Uri': 's3://omm-test-bucket/models/abalone/data/input/test/test_X.csv'}},
#     'ContentType': 'text/csv',
#     'SplitType': 'Line'},
#     'TransformOutput': {'S3OutputPath': 's3://omm-test-bucket/models/test/data/transformations/out',
#     'Accept': 'text/csv',
#     'AssembleWith': 'Line'},
#     'TransformResources': {'InstanceType': 'ml.m5.large', 'InstanceCount': 1}
# }
# monitoring_job_definition=self.get_job_definition(self.sagemaker_session, self.mb_monitor_dir, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None)
# model_dashboard_monitoring_schedule=ModelDashboardMonitoringSchedule(
#     batch_transform_input=transform_step.arguments['TransformInput'],
#     monitoring_schedule_config=MonitoringScheduleConfig(
#         schedule_config=schedule_config,
#         monitoring_job_definition_name="ModelBiasJobDefinition",
#         monitoring_job_definition=monitoring_job_definition,
#         monitoring_type="ModelBias" #DataQuality | ModelQuality | ModelBias | ModelExplainability
#     )
# )  

class MonitorMaker():
    def __init__(self, model_name, data_dir_uri, baseline_file, train_file, monitor_instance_type, sagemaker_session=None):
        self.model_name=model_name
        self.baseline_file= baseline_file # f'{data_dir_uri}/baseline/baseline.csv'
        self.train_file=    train_file   # f'{data_dir_uri}/input/train/train.csv'
        self.train_X_file=  f'{data_dir_uri}/monitors/model-explainability/train_X.csv'
        self.baseline_pred_file=f'{data_dir_uri}/baseline/baseline_pred.csv'
        self.dq_monitor_dir=f'{data_dir_uri}/monitors/data-quality'
        self.mq_monitor_dir=f'{data_dir_uri}/monitors/model-quality'
        self.mb_monitor_dir=f'{data_dir_uri}/monitors/model-bias'
        self.me_monitor_dir=f'{data_dir_uri}/monitors/model-explainability'
        self.mbt_monitor_dir=f'{data_dir_uri}/monitors/batch'
        self.data_capture_dir=  f'{data_dir_uri}/capture'
        self.ground_truth_dir=  f'{data_dir_uri}/ground-truth'
        
        self.monitor_instance_type=monitor_instance_type
        self.sagemaker_session=sagemaker_session
    
    def get_monitor_batch_transform_step(self, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None, depends_on=[]):


        batch_transform_input = BatchTransformInput(
            data_captured_destination_s3_uri=f'{self.data_capture_dir}',
            dataset_format=MonitoringDatasetFormat(csv=MonitoringCsvDatasetFormat(header=True)),
            local_path='/opt/ml/processing/input',
            s3_input_mode='File',
            s3_data_distribution_type='FullyReplicated', 
            features_attribute=','.join(self.features),
            inference_attribute=self.target,
            probability_attribute=probability_attribute,
            probability_threshold_attribute=probability_threshold_attribute,
            start_time_offset="-PT2H",
            end_time_offset="-PT1H",
            exclude_features_attribute=exclude_features_attribute,
            sagemaker_session=self.sagemaker_session
        )

        monitoring_resources = MonitoringResources(
            cluster_config=MonitoringClusterConfig(
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=5,
                volume_kms_key_id=None
            )
        )
        
        monitoring_app_specification=MonitoringAppSpecification(
            image_uri=retrieve_image(framework='model-monitor', region='us-east-1'), # required - the container to run
            # container_entrypoint=['...'],       # optional - override entrypoint
            # container_arguments=['...'],        # optional - override arguments
            # record_preprocessor_source_uri='s3://...', # optional - preprocessing script
            # post_analytics_processor_source_uri='s3://...' # optional - postprocessing script
        )

        monitoring_job_definition = MonitoringJobDefinition(
            monitoring_inputs= [MonitoringInput(batch_transform_input=batch_transform_input)], 
            monitoring_output_config=MonitoringOutputConfig(
                monitoring_outputs=[
                    MonitoringOutput(
                        s3_output=MonitoringS3Output(
                            local_path='/opt/ml/processing/input', 
                            s3_uri='s3://omm-test-bucket/models/test/batch-output/'
                        )
                    )
                ]
            ),
            monitoring_resources=monitoring_resources, 
            monitoring_app_specification=monitoring_app_specification, 
            role_arn=get_execution_role(), 
            stopping_condition=MonitoringStoppingCondition(
                max_runtime_in_seconds=400
            ), 
            environment={}, 
            # network_config: NetworkConfig | None = Unassigned()
        )

        schedule_config = ScheduleConfig(
            schedule_expression='cron(0 * ? * * *)', 
            data_analysis_start_time="-PT1H", 
            data_analysis_end_time="-PT2H"
            )

#################### MODEL QUALITY

    def get_monitor_batch_transform_step(self, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None, depends_on=[]):

        batch_transform_input = BatchTransformInput(
            data_captured_destination_s3_uri=f'{self.data_capture_dir}',
            dataset_format=MonitoringDatasetFormat(csv=MonitoringCsvDatasetFormat(header=True)),
            local_path='/opt/ml/processing/input',
            s3_input_mode='File',
            s3_data_distribution_type='FullyReplicated', 
            features_attribute=','.join(self.features),
            inference_attribute=self.target,
            probability_attribute=probability_attribute,
            probability_threshold_attribute=probability_threshold_attribute,
            start_time_offset="-PT2H",
            end_time_offset="-PT1H",
            exclude_features_attribute=exclude_features_attribute,
            sagemaker_session=self.sagemaker_session
        )

        monitoring_resources = MonitoringResources(
            cluster_config=MonitoringClusterConfig(
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=5,
                volume_kms_key_id=None
            )
        )

        return [batch_transform_input, monitoring_resources]
    
    def get_job_definition(self, monitoring_output, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None):

        batch_transform_input, monitoring_resources=self.get_monitor_batch_transform_step(self.sagemaker_session)
                                         
        monitoring_app_specification=MonitoringAppSpecification(
            image_uri=retrieve_image(framework='model-monitor', region='us-east-1'), # required - the container to run
            # container_entrypoint=['...'],       # optional - override entrypoint
            # container_arguments=['...'],        # optional - override arguments
            # record_preprocessor_source_uri='s3://...', # optional - preprocessing script
            # post_analytics_processor_source_uri='s3://...' # optional - postprocessing script
        )

        monitoring_job_definition = MonitoringJobDefinition(
            monitoring_inputs= [MonitoringInput(batch_transform_input=batch_transform_input)], 
            monitoring_output_config=MonitoringOutputConfig(
                monitoring_outputs=[
                    MonitoringOutput(
                        s3_output=MonitoringS3Output(
                            local_path='/opt/ml/processing/input', 
                            s3_uri=monitoring_output
                        )
                    )]
            ),
            monitoring_resources=monitoring_resources, 
            monitoring_app_specification=monitoring_app_specification, 
            role_arn=get_execution_role(), 
            stopping_condition=MonitoringStoppingCondition(
                max_runtime_in_seconds=400
            ), 
            environment={}, 
            # network_config: NetworkConfig | None = Unassigned()
        )
    
        return monitoring_job_definition

        schedule_config = ScheduleConfig(
            schedule_expression='cron(0 * ? * * *)', 
            data_analysis_start_time="-PT1H", 
            data_analysis_end_time="-PT2H"
            )

    def get_batch_model_quality_step(self, transform_step, schedule_config, depends_on=[]):

        monitoring_job_definition=self.get_job_definition(self.sagemaker_session, self.mq_monitor_dir, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None)

        model_dashboard_monitoring_schedule=ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="'ModelQualityJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="ModelQuality" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )

        model_quality_check_config=ModelQualityCheckConfig(
            baseline_dataset=f'{self.mq_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            problem_type='Regression',
            output_s3_uri=f'{self.mq_monitor_dir}/info'
        )
        
        mq_monitor_step = MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=model_quality_check_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.mq_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.mq_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.mq_monitor_dir}/reports',
            ground_truth_input=f'{self.ground_truth_dir}/',  # ground truth labels
            fail_on_violation=False,
            depends_on=depends_on,
            sagemaker_session=self.sagemaker_session
        )

        return mq_monitor_step

#################### MODEL BIAS
    def get_batch_model_bias_step(self, transform_step, role, schedule_config, depends_on=[]):
    
        model_bias_check_config=ModelQualityCheckConfig(
            baseline_dataset=f'{self.mb_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            problem_type='Regression',
            output_s3_uri=f'{self.mb_monitor_dir}/predictions'
        )
        
        mb_monitor_step = MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=model_bias_check_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.mb_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.mb_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.mb_monitor_dir}/reports',
            ground_truth_input='s3://omm-test-bucket/models/abalone/data/ground-truth/',  # ground truth labels
            fail_on_violation=False,
            sagemaker_session=self.sagemaker_session
        )
        return mb_monitor_step

#################### DATA QUALITY
    def get_batch_data_quality_step(self, transform_step, role, schedule_config, depends_on=[]):

        monitoring_job_definition=self.get_job_definition(self.sagemaker_session, self.dq_monitor_dir, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None)

        model_dashboard_monitoring_schedule=ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="DataQualityJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="DataQuality" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )
              
        data_quality_config=DataQualityCheckConfig(
                baseline_dataset=f'{self.dq_monitor_dir}/baseline.csv', 
                dataset_format={}, 
                output_s3_uri=f'{self.dq_monitor_dir}/predictions'
        )
        
        dq_monitor_step = MonitorBatchTransformStep(
            name='DataQualityMonitorStep',
            monitor_configuration=data_quality_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.dq_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.dq_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.dq_monitor_dir}/reports',
            ground_truth_input=f'{self.ground_truth_dir}/',  # ground truth labels
            fail_on_violation=False,
            sagemaker_session=self.sagemaker_session
        )

        return dq_monitor_step

#################### DATA BIAS
    def get_batch_data_bias_step(self, transform_step, schedule_config, depends_on=[]):

        monitoring_job_definition=self.get_job_definition(self.sagemaker_session, self.db_monitor_dir, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None)

        model_dashboard_monitoring_schedule=ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="DataBiasJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="DataBias" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )
                
        data_bias_check_config=ModelQualityCheckConfig(
            baseline_dataset=f'{self.db_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            problem_type='Regression',
            output_s3_uri=f'{self.db_monitor_dir}/predictions'
        )
        
        mq_monitor_step = MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=data_bias_check_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.db_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.db_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.db_monitor_dir}/reports',
            ground_truth_input=f'{self.ground_truth_dir}/',  # ground truth labels
            fail_on_violation=False,
            sagemaker_session=self.sagemaker_session
        )

#################### EXPLAINABILITY
    def get_batch_model_explainabilty_step(self, transform_step, role, schedule_config, depends_on=[]):

        monitoring_job_definition=self.get_job_definition(self.sagemaker_session, self.me_monitor_dir, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None)

        model_dashboard_monitoring_schedule=ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="ModelExplainabilityJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="ModelExplainability" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )
                
        model_explainabilty_check_config=ModelQualityCheckConfig(
            baseline_dataset=f'{self.me_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            problem_type='Regression',
            output_s3_uri=f'{self.me_monitor_dir}/predictions'
        )
        
        me_monitor_step = MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=model_explainabilty_check_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.me_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.me_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.me_monitor_dir}/reports',
            ground_truth_input=f'{self.ground_truth_dir}/',  # ground truth labels
            fail_on_violation=False,
            sagemaker_session=self.sagemaker_session
        )

        return me_monitor_step