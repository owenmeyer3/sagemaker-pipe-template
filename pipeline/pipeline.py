from sagemaker.core.workflow.pipeline_context import PipelineSession, LocalPipelineSession
from sagemaker.core.workflow.parameters import ParameterString, ParameterInteger, ParameterBool
from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.image_uris import retrieve as retrieve_image
from sagemaker.core.resources import Model
from sagemaker.core.lambda_helper import Lambda
from sagemaker.core.transformer import Transformer
from sagemaker.core.inputs import TransformInput
from sagemaker.core.shapes.shapes import ScheduleConfig, ContainerDefinition
from sagemaker.core.workflow.conditions import ConditionEquals
from sagemaker.core.workflow.functions import Join

from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.steps import TransformStep
from sagemaker.mlops.workflow.lambda_step import LambdaOutputTypeEnum, LambdaStep, LambdaOutput
from sagemaker.mlops.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.mlops.workflow.condition_step import ConditionStep
import utils, baseline, boto3, argparse, monitor, json, paths, datetime, time, steps
import pandas as pd


class CPipeline(Pipeline):
    def __init__(self, 
        sagemaker_session, 
        name, 
        pipeline_bucket,
        deployment_type, 
        target_name, 
        target_type, 
        problem_type,
        prediction_name, 
        model_package_group_name,
        region_name, 
        lambda_execution_role_arn, 
        other_execution_role_arn
    ):

        self.sagemaker_session = sagemaker_session
        self.region_name = region_name
        self.name = name
        self.pipeline_bucket=pipeline_bucket
        self.deployment_type=deployment_type
        self.target_name=target_name
        self.target_type = target_type
        self.problem_type=problem_type
        self.prediction_name_param=prediction_name
        self.model_package_group_name=model_package_group_name
        self.lambda_execution_role_arn=lambda_execution_role_arn
        self.other_execution_role_arn=other_execution_role_arn

        self.model_package_version_param =                  ParameterInteger(name='ModelPackageVersion',                default_value=1)
        self.action_param =                                 ParameterString( name='Action',                                                 enum_values=['deploy', 'inference'])
        self.baseline_file_param =                          ParameterString( name='BaselineFile')
        self.monitor_instance_type_param =                  ParameterString( name='MonitorInstanceType',                 default_value='ml.m5.large')
        self.endpoint_instance_type_param =                 ParameterString( name='EndpointInstanceType',                default_value='ml.m5.large')
        self.transform_instance_type_param =                ParameterString( name='TransformInstanceType',               default_value='ml.m5.large')
        self.fail_on_violation_param =                      ParameterBool(   name='FailOnViolation',                     default_value=False)
        self.register_new_baseline_param =                  ParameterBool(   name='RegisterNewBaseline',                 default_value=False)
        self.schedule_expression_param =                    ParameterString( name='MonitorScheduleExpression',           default_value='cron(0 * ? * * *)')
        self.enable_data_quality_monitoring_param =         ParameterBool(   name='EnableDataQualityMonitoring',         default_value=True)
        self.enable_model_bias_monitoring_param =           ParameterBool(   name='EnableModelBiasMonitoring',           default_value=True)
        self.enable_model_explainability_monitoring_param = ParameterBool(   name='EnableModelExplainabilityMonitoring', default_value=True)
        self.enable_model_quality_monitoring_param =        ParameterBool(   name='EnableModelQualityMonitoring',        default_value=True)
        self.environment_param =                            ParameterString( name='Environment',                         default_value='dev', enum_values=['prd', 'dev', 'stg'])
        self.sns_topic_arn_param =                          ParameterString( name='SnsTopicArn',                         default_value='')
        self.enable_sns_notification_param =                ParameterBool(   name='EnableSnsNotification',               default_value=False)
        self.ground_truth_dir_param =                       ParameterString( name='GroundTruthDir',                      default_value=f's3://{pipeline_bucket}/ground-truth/{model_package_group_name}')
        self.batch_input_dir_param =                        ParameterString( name='BatchInputDir',                       default_value=f's3://{pipeline_bucket}/batch_input/{model_package_group_name}')


        self.pipeline_dir = f's3://{pipeline_bucket}/pipelines/{model_package_group_name}'
        self.baseline_dir = f'{self.pipeline_dir}/baseline'
        self.monitors_dir=  f'{self.pipeline_dir}/monitors'
        self.batch_out_dir=f'{self.pipeline_dir}/batch_out',
        self.data_capture_dir=f'{self.pipeline_dir}/capture',
        self.dq_monitor_dir=f'{self.pipeline_dir}/data-quality',
        self.mq_monitor_dir=f'{self.pipeline_dir}/model-quality',
        self.mb_monitor_dir=f'{self.pipeline_dir}/model-bias',
        self.me_monitor_dir=f'{self.pipeline_dir}/model-explainability',
        # self.paths={
        #     'pipeline_dir': pipeline_dir,
        #     'baseline_dir': baseline_dir,
        #     'monitors_dir': monitors_dir,
        #     'batch_out_dir': f'{pipeline_dir}/batch_out',
        #     'data_capture_dir': f'{pipeline_dir}/capture',
        #     'dq_monitor_dir': f'{pipeline_dir}/data-quality',
        #     'mq_monitor_dir': f'{pipeline_dir}/model-quality',
        #     'mb_monitor_dir': f'{pipeline_dir}/model-bias',
        #     'me_monitor_dir': f'{pipeline_dir}/model-explainability',
        # }

        self.p_params = paths.PathParams(self.training_bucket_param, self.training_dir_param, self.pipeline_bucket_param, self.name)

        ###########################
        ### GET / CREATE MODEL ####
        ###########################
        get_or_create_model_from_registry_step = steps.GetOrCreateModelFromRegistryStep(
            name='GetOrCreateModelFromRegistry',
            function_name=f'GetOrCreateModelFromRegistry-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            model_package_group_name_param=self.model_package_group_name,
            model_package_version_param=self.model_package_version_param,
            sagemaker_session=self.sagemaker_session,
            depends_on=[]
        )
        self.model_name_param = get_or_create_model_from_registry_step.properties.Outputs['model_name']
        self.model_package_arn_param = get_or_create_model_from_registry_step.properties.Outputs['model_package_arn']

        ###########################
        ######## BASELINE #########
        ###########################
        prep_baseline_step=steps.PrepBaselineSetsStep(
            name='PrepBaselineSetsStep',
            function_name=f'PrepBaselineSetsStep-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            baseline_file=self.baseline_file_param,
            target_name=self.target_name,
            target_type=self.target_type,
            baseline_X_file_dest_dir=self.baseline_dir,
            sagemaker_session=self.sagemaker_session,
            depends_on=[]
        )
        self.baseline_X_dir = prep_baseline_step.properties.Outputs['baseline_X_dir']
        self.baseline_X_filename = prep_baseline_step.properties.Outputs['baseline_X_filename']

        # transform
        baseline_transform_step = TransformStep(
            name="TransformStep",
            transformer=Transformer(
                model_name=self.model_name,
                instance_count=1,
                instance_type=self.transform_instance_type_param,
                output_path=self.transform_out_dir,
                accept='text/csv',
                assemble_with='Line'
            ),
            inputs=TransformInput(data=self.baseline_X_filename, content_type='text/csv', split_type='Line'),
            sagemaker_session=self.sagemaker_session,
            depends_on=[prep_baseline_step]
        )

        get_baseline_preds_step=steps.GetBaselinePredsStep(
            name='GetBaselinePredsStep',
            function_name=f'GetBaselinePredsStep-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            transform_out_dir=self.baseline_dir,
            baseline_X_filename=self.baseline_X_filename,
            baseline_pred_file_dest=self.baseline_dir,
            sagemaker_session=self.sagemaker_session,
            depends_on=[baseline_transform_step]
        )
        self.baseline_pred_file = get_baseline_preds_step.properties.Outputs['baseline_pred_file']

        make_baseline_step = steps.MakeBaselineSetsStep(
            name='MakeBaselineSetsStep',
            function_name=f'MakeBaselineSetsStep-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            target_name=self.target_name,
            prediction_name=self.prediction_name,
            target_type=self.target_type,
            baseline_file=self.baseline_file_param,
            baseline_pred_file=self.baseline_pred_file,
            dq_monitor_dir=self.dq_monitor_dir,
            db_monitor_dir=self.db_monitor_dir,
            mq_monitor_dir=self.mq_monitor_dir,
            mb_monitor_dir=self.mb_monitor_dir,
            me_monitor_dir=self.me_monitor_dir,
            baseline_X_file=f'{self.baseline_X_file}/{self.baseline_X_filename}',
            sagemaker_session=self.sagemaker_session,
            depends_on=[get_baseline_preds_step]
        )

        ###########################
        ######## DEPLOY ###########
        ###########################
        deploy_endpoint_step=steps.DeployEndpointStep(
            name='DeployEndpointStep',
            function_name=f'DeployEndpointStep-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            model_name_param=get_or_create_model_from_registry_step.properties.Outputs['model_name'],
            model_package_group_name_param=self.model_package_group_name,
            model_package_version_param=self.model_package_version_param,
            data_capture_dir=self.data_capture_dir,
            sagemaker_session=self.sagemaker_session,
            depends_on=[]
        )
        self.endpoint_name_param = deploy_endpoint_step.properties.Outputs['endpoint_name']

        ###########################
        ##### SCHEDULE MONS #######
        ###########################
        create_schedules_dq_monitor_step=steps.CreateScheduledDataQualityMonitorStep(
            name='ScheduledDataQualityMonitorStep',
            function_name=f'ScheduledDataQualityMonitorStep-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            deploy_type=self.deploy_type,
            monitor_dir=self.dq_monitor_dir,
            monitor_name=f'DataQualityMonitor-{self.model_package_group_name}',
            endpoint_name=self.endpoint_name_param,
            data_cature_dir=self.data_cature_dir,
            image_uri = "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
            instance_count = 1,
            instance_type = self.monitor_instance_type_param,
            volume_size_in_gb = 20,
            max_runtime_in_seconds = 1800,
            dataset_format = {'Csv': {'Header': True}},
            schedule_expression = 'cron(0 * ? * * *)',
            data_analysis_start_time = "-PT2H",
            data_analysis_end_time = "-PT1H",
            sagemaker_session=self.sagemaker_session,
            depends_on=[]
        )

        create_schedules_mb_monitor_step=steps.CreateScheduledModelBiasMonitorStep(
            name='ScheduledModelBiasMonitorStep',
            function_name=f'ScheduledModelBiasMonitorStep-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            deploy_type=self.deploy_type,
            monitor_dir=self.mb_monitor_dir,
            ground_truth_dir=self.ground_truth_dir_param,
            monitor_name=f'ModelBiasMonitor-{self.model_package_group_name}',
            endpoint_name=self.endpoint_name_param,
            data_cature_dir=self.data_cature_dir,
            image_uri = "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
            instance_count = 1,
            instance_type = self.monitor_instance_type_param,
            volume_size_in_gb = 20,
            max_runtime_in_seconds = 1800,
            dataset_format = {'Csv': {'Header': True}},
            schedule_expression = 'cron(0 * ? * * *)',
            data_analysis_start_time = "-PT2H",
            data_analysis_end_time = "-PT1H",
            sagemaker_session=self.sagemaker_session,
            depends_on=[]
        )

        create_schedules_me_monitor_step=steps.CreateScheduledModelExplainabilityMonitorStep(
            name='ScheduledModelExplainabilityMonitorStep',
            function_name=f'ScheduledModelExplainabilityMonitorStep-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            deploy_type=self.deploy_type,
            monitor_dir=self.me_monitor_dir,
            monitor_name=f'ModelExplainabilityMonitor-{self.model_package_group_name}',
            endpoint_name=self.endpoint_name_param,
            data_cature_dir=self.data_cature_dir,
            image_uri = "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
            instance_count = 1,
            instance_type = self.monitor_instance_type_param,
            volume_size_in_gb = 20,
            max_runtime_in_seconds = 1800,
            dataset_format = {'Csv': {'Header': True}},
            schedule_expression = 'cron(0 * ? * * *)',
            data_analysis_start_time = "-PT2H",
            data_analysis_end_time = "-PT1H",
            sagemaker_session=self.sagemaker_session,
            depends_on=[]
        )

        create_schedules_mq_monitor_step=steps.CreateScheduledModelQualityMonitorStep(
            name='ScheduledModelQualityMonitorStep',
            function_name=f'ScheduledModelQualityMonitorStep-{self.model_package_group_name}',
            execution_role_arn=self.lambda_execution_role_arn,
            deploy_type=self.deploy_type,
            problem_type=self.problem_type,
            ground_truth_label=self.ground_truth_label,
            monitor_dir=self.mq_monitor_dir,
            ground_truth_dir=self.ground_truth_dir_param,
            monitor_name=f'ModelQualityMonitor-{self.model_package_group_name}',
            endpoint_name=self.endpoint_name_param,
            data_cature_dir=self.data_cature_dir,
            image_uri = "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
            instance_count = 1,
            instance_type = self.monitor_instance_type_param,
            volume_size_in_gb = 20,
            max_runtime_in_seconds = 1800,
            dataset_format = {'Csv': {'Header': True}},
            schedule_expression = 'cron(0 * ? * * *)',
            data_analysis_start_time = "-PT2H",
            data_analysis_end_time = "-PT1H",
            sagemaker_session=self.sagemaker_session,
            depends_on=[]
        )
        ###########################
        ###########################
        ###########################

        ## Choice
        is_inference = ConditionEquals(left=self.action_param, right='inference')
        is_register_new_baseline = ConditionEquals(left=self.register_new_baseline_param, right=True)
        is_data_quality_monitoring = ConditionEquals(left=self.enable_data_quality_monitoring_param, right=True)
        is_model_bias_monitoring = ConditionEquals(left=self.enable_model_bias_monitoring_param, right=True)
        is_model_explainability_monitoring = ConditionEquals(left=self.enable_model_explainability_monitoring_param, right=True)
        is_model_quality_monitoring = ConditionEquals(left=self.enable_model_quality_monitoring_param, right=True)
        is_sns_notification = ConditionEquals(left=self.enable_sns_notification_param, right=True)


        baseline_or_not_steps = ConditionStep(name='BaselineChoice', conditions=[is_register_new_baseline], 
            if_steps=prep_baseline_step, 
            else_steps=[], 
            depends_on=get_or_create_model_from_registry_step
        )

        deploy_or_inference_steps = ConditionStep(name='ActionTypeChoice', conditions=[is_inference], 
            if_steps=deploy_endpoint_step, 
            # else_steps=inference_step, 
            depends_on=[baseline_or_not_steps]
        )

        data_quality_or_not_steps = ConditionStep(name='DataQualityChoice', conditions=[is_data_quality_monitoring], 
            if_steps=[create_schedules_dq_monitor_step], 
            else_steps=[], 
            depends_on=[]
        )
        model_bias_or_not_steps = ConditionStep(name='ModelBiasChoice', conditions=[is_model_bias_monitoring], 
            if_steps=[create_schedules_mb_monitor_step], 
            else_steps=[], 
            depends_on=[data_quality_or_not_steps]
        )
        model_explainability_or_not_steps = ConditionStep(name='ModelExplainabilityChoice', conditions=[is_model_explainability_monitoring], 
            if_steps=[create_schedules_me_monitor_step], 
            else_steps=[], 
            depends_on=[model_bias_or_not_steps]
        )
        model_quality_or_not_steps = ConditionStep(name='ModelQualityChoice', conditions=[is_model_quality_monitoring], 
            if_steps=[create_schedules_mq_monitor_step], 
            else_steps=[], 
            depends_on=[model_explainability_or_not_steps]
        )
        sns_notification_or_not_steps = ConditionStep(name='SnsNotificationChoice', conditions=[is_sns_notification], 
            if_steps=[], 
            else_steps=[], 
            depends_on=[]
        )

        ###########################
        ## Full Pipe
        steps = [get_or_create_model_from_registry_step]
        super().__init__(
            name=self.name, 
            parameters=self.parameters, 
            steps=[], 
            sagemaker_session=self.sagemaker_session
        )

        ###########################

def create_pipe(
        name, 
        deployment_type, 
        target_name, 
        target_type, 
        prediction_name, 
        problem_type,
        region_name='us-east-1',
        lambda_execution_role_arn='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1',
        other_execution_role_arn='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1'
    ):

    sagemaker_session = LocalPipelineSession(boto_session=boto3.Session(region_name=region_name))
    # sagemaker_session = PipelineSession(boto_session=boto3.Session(region_name=region_name))
    invoke_steps_role = get_execution_role(sagemaker_session)
    
    pipeline = CPipeline(
        sagemaker_session, 
        name=name,
        deployment_type=deployment_type,
        target_name=target_name,
        target_type=target_type,
        problem_type=problem_type,
        prediction_name=prediction_name,
        lambda_execution_role_arn=lambda_execution_role_arn,
        other_execution_role_arn=other_execution_role_arn
    )
    pipeline_definition = json.loads(pipeline.definition())
    print(json.dumps(pipeline_definition, indent=2))
    pipeline.upsert(role_arn=invoke_steps_role)
    


def run(pipeline_name, wait=False, pipeline_parameters={}):

    # pipeline_parameters=[
    #     {'Name': 'InputData', 'Value': 's3://omm-test-bucket/data/test.csv'},
    #     {'Name': 'InstanceType', 'Value': 'ml.m5.large'}
    # ]

    sm_client = boto3.client('sagemaker', region_name='us-east-1')

    ex_response = sm_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=f"{pipeline_name}-execution-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        # Override
        PipelineParameters=pipeline_parameters
    )
    ex_arn=ex_response['PipelineExecutionArn']

    if not wait:
        print(f'started pipe execution {ex_arn}')
        return


    while True:
        ex_details = sm_client.describe_pipeline_execution(PipelineExecutionArn=ex_arn)
        status = ex_details['PipelineExecutionStatus']
        print(f"Status: {status}")
        if status in ['Succeeded', 'Failed', 'Stopped']:
            break
        time.sleep(30)

def test_main():
    sm_client = boto3.client('sagemaker', region_name='us-east-1')

    # List all pipelines
    response = sm_client.list_pipelines()
    for pipeline in response['PipelineSummaries']:
        print(pipeline['PipelineName'])
    
    # delete monitors and endpoint
    import boto3
    sm_client = boto3.client('sagemaker', region_name='us-east-1')
    endpoint_name='abalone-endpoint'
    # List and delete monitoring schedules
    schedules = sm_client.list_monitoring_schedules(
        EndpointName=endpoint_name
    )

    for schedule in schedules['MonitoringScheduleSummaries']:
        sm_client.delete_monitoring_schedule(
            MonitoringScheduleName=schedule['MonitoringScheduleName']
        )
        print(f"Deleted schedule: {schedule['MonitoringScheduleName']}")

    # Then delete endpoint
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print("Endpoint deleted")


    exit()


if __name__ == '__main__':
    # python3 pipeline/pipeline.py --action deploy --deployment-type realtime --model-package-group-name abalone --model-package-version 1 --pipe-name sagemaker-pipe-template --target-name rings --prediction-name rings_prediction --project-bucket omm-test-bucket --project-path "models/abalone" --monitor-instance-type "ml.m5.large"
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--action', type=str, default='deploy')
    # args = parser.parse_args()
    create_pipe(
        name='abalone-pipe', 
        deployment_type='realtime', 
        target_name='rings', 
        target_type='float', 
        prediction_name='rings-prediction', 
        region_name='us-east-1',
        lambda_execution_role_arn='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1',
        other_execution_role_arn='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1'
    )

    # run(
    #     'abalone-pipe', 
    #     wait=False, 
    #     pipeline_parameters=[
    #         {'Name': 'ModelPackageGroupName', 'Value': 'abalone'},
    #         {'Name': 'ModelPackageVersion', 'Value': 1},
    #         {'Name': 'RuntimeRole', 'Value': 'arn:aws:iam::088461143167:role/SageMakerExecutionRole-1'},
    #         {'Name': 'Action', 'Value': 'deploy'},
    #         {'Name': 'ProjectBucket', 'Value': 'abalone'},
    #         {'Name': 'ProjectPath', 'Value': 'abalone'},
    #         {'Name': 'MonitorInstanceType', 'Value': 'ml.m5.large'},
    #         {'Name': 'EndpointInstanceType', 'Value': 'ml.m5.large'},
    #         {'Name': 'TransformInstanceType', 'Value': 'ml.m5.large'},
    #         {'Name': 'TrainingBucket', 'Value': 'omm-test-bucket'},
    #         {'Name': 'TrainingDir', 'Value': 'projects/abalone/models/abalone'},
    #         {'Name': 'PipelineBucket', 'Value': 'omm-test-bucket'},
    #     ]
    # )