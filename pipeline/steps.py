from sagemaker.core.lambda_helper import Lambda
from sagemaker.mlops.workflow.lambda_step import LambdaOutputTypeEnum, LambdaStep, LambdaOutput
from sagemaker.core.helper.session_helper import Session
from typing import List, Dict, Optional


class LambdaStepBase(LambdaStep):
    def __init__(self,
        name: str,
        function_name: str,
        execution_role_arn: str,
        script: str = None,
        handler: str = None,
        inputs: Optional[dict] = None,
        outputs: Optional[List[LambdaOutput]] = None,
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):
        lambda_kwargs = lambda_kwargs or {}

        lambda_func = Lambda(
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script=script,
            handler=handler,
            session=sagemaker_session,
            **lambda_kwargs # function_arn, zipped_code_dir, s3_bucket, timeout, memory_size, runtime, vpc_config, environment, layers
        )

        super().__init__(
            name=name,
            lambda_func=lambda_func,
            inputs=inputs,
            outputs=outputs,
            **step_kwargs # display_name, description, cache_config, vpc_config, environment, layers, depends_on
        )


class GetOrCreateModelFromRegistryStep(LambdaStepBase):
    def __init__(self, 
        name: str,
        function_name: str,
        execution_role_arn: str,
        model_package_group_name_param: str,
        model_package_version_param: str,
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):
        inputs={
            'model_package_group_name': model_package_group_name_param,
            'model_package_version': model_package_version_param
        }

        outputs=[
            LambdaOutput(output_name='model_name', output_type=LambdaOutputTypeEnum.String),
            LambdaOutput(output_name='model_package_arn', output_type=LambdaOutputTypeEnum.String)
        ]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/get_or_create_model_from_registry.py',  # path to your file
            handler='get_or_create_model_from_registry.handler',    # filename.function_name
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )


class DeployEndpointStep(LambdaStepBase):
    def __init__(self, 
        name: str,
        function_name: str,
        execution_role_arn: str,
        model_name_param: str,
        model_package_group_name: str,
        model_package_version_param: str,
        data_capture_dir:str,
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):
        inputs={
            'model_name': model_name_param,
            'model_package_group_name': model_package_group_name,
            'model_package_version_param': model_package_version_param,
            'data_capture_dir':data_capture_dir
        }

        outputs=[
            LambdaOutput(output_name='endpoint_name', output_type=LambdaOutputTypeEnum.String)
        ]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/deploy_endpoint.py',
            handler='deploy_endpoint.handler',
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )


class PrepBaselineSetsStep(LambdaStepBase):
    def __init__(self, 
        name: str,
        function_name: str,
        execution_role_arn: str,
        baseline_file: str,
        target_name: str,
        target_type: str,
        baseline_X_file_dest_dir:str,
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):
        inputs={
            'baseline_file': baseline_file,
            'target_name':target_name,
            'target_type':target_type,
            'baseline_X_file_dest_dir':baseline_X_file_dest_dir
        }

        outputs=[
            LambdaOutput(output_name='baseline_X_dir', output_type=LambdaOutputTypeEnum.String),
            LambdaOutput(output_name='baseline_X_filename', output_type=LambdaOutputTypeEnum.String),
        ]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/baselining.py',
            handler='baselining.prep_baseline_sets_handler',
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )


class GetBaselinePredsStep(LambdaStepBase):
    def __init__(self, 
        name: str,
        function_name: str,
        execution_role_arn: str,
        transform_out_dir: str,
        baseline_X_filename: str,
        baseline_pred_file_dest: str,
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):
        inputs={
            'transform_out_dir': transform_out_dir,
            'baseline_X_filename':baseline_X_filename,
            'baseline_pred_file_dest':baseline_pred_file_dest
        }

        outputs=[
            LambdaOutput(output_name='baseline_pred_file', output_type=LambdaOutputTypeEnum.String),
        ]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/baselining.py',
            handler='baselining.get_baseline_preds_handler',
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )


class MakeBaselineSetsStep(LambdaStepBase):
    def __init__(self, 
        name: str,
        function_name: str,
        execution_role_arn: str,
        target_name: str,
        prediction_name: str,
        target_type: str,
        baseline_file: str,
        baseline_pred_file: str,
        dq_monitor_dir: str,
        db_monitor_dir: str,
        mq_monitor_dir: str,
        mb_monitor_dir: str,
        me_monitor_dir: str,
        train_file: str,
        train_X_file: str,
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):
        inputs={
            'target_name':target_name,
            'prediction_name':prediction_name,
            'target_type':target_type,
            'baseline_file':baseline_file,
            'baseline_pred_file':baseline_pred_file,
            'dq_monitor_dir':dq_monitor_dir,
            'db_monitor_dir':db_monitor_dir,
            'mq_monitor_dir':mq_monitor_dir,
            'mb_monitor_dir':mb_monitor_dir,
            'me_monitor_dir':me_monitor_dir,
            'train_file':train_file,
            'train_X_file':train_X_file
        }

        outputs=[]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/baselining.py',
            handler='baselining.make_baseline_sets_handler',
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )


class CreateScheduledDataQualityMonitorStep(LambdaStepBase):
    def __init__(self, 
        name:str,
        function_name:str,
        execution_role_arn:str,
        deploy_type:str,
        monitor_dir:str,
        monitor_name:str,
        endpoint_name:str,
        data_cature_dir:str,
        image_uri:str = "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
        instance_count:int = 1,
        instance_type:str = 'ml.m5.large',
        volume_size_in_gb:str = 20,
        max_runtime_in_seconds:str = 1800,
        dataset_format:str = {'Csv': {'Header': True}}, # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}} 
        schedule_expression:str = 'cron(0 * ? * * *)',
        data_analysis_start_time:str = "-PT2H",
        data_analysis_end_time:str = "-PT1H",
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):

        inputs={
            'name': monitor_name,
            'role_arn': execution_role_arn,
            'deploy_type': deploy_type,
            'monitor_dir': monitor_dir,
            'image_uri': image_uri,
            'instance_count': instance_count,
            'instance_type': instance_type,
            'volume_size_in_gb': volume_size_in_gb,
            'max_runtime_in_seconds': max_runtime_in_seconds,
            'dataset_format': dataset_format,
            'schedule_expression': schedule_expression,
            'data_analysis_start_time': data_analysis_start_time,
            'data_analysis_end_time': data_analysis_end_time,
            'endpoint_name': endpoint_name,
            'data_cature_dir': data_cature_dir,
        }

        outputs=[]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/schedule_monitors.py',
            handler='schedule_monitors.data_quality_handler',
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )


class CreateScheduledModelBiasMonitorStep(LambdaStepBase):
    def __init__(self, 
        name: str,
        function_name: str,
        execution_role_arn: str,
        deploy_type:str,
        monitor_dir:str,
        ground_truth_dir:str,
        monitor_name:str,
        endpoint_name:str,
        data_cature_dir:str,
        image_uri:str = "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
        instance_count:int = 1,
        instance_type:str = 'ml.m5.large',
        volume_size_in_gb:str = 20,
        max_runtime_in_seconds:str = 1800,
        dataset_format:str = {'Csv': {'Header': True}}, # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}} 
        schedule_expression:str = 'cron(0 * ? * * *)',
        data_analysis_start_time:str = "-PT2H",
        data_analysis_end_time:str = "-PT1H",
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):

        inputs={
            'name': monitor_name,
            'role_arn': execution_role_arn,
            'deploy_type': deploy_type,
            'monitor_dir': monitor_dir,
            'ground_truth_dir': ground_truth_dir,
            'image_uri': image_uri,
            'instance_count': instance_count,
            'instance_type': instance_type,
            'volume_size_in_gb': volume_size_in_gb,
            'max_runtime_in_seconds': max_runtime_in_seconds,
            'dataset_format': dataset_format,
            'schedule_expression': schedule_expression,
            'data_analysis_start_time': data_analysis_start_time,
            'data_analysis_end_time': data_analysis_end_time,
            'endpoint_name': endpoint_name,
            'data_cature_dir': data_cature_dir,
        }

        outputs=[]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/schedule_monitors.py',
            handler='schedule_monitors.model_bias_handler',
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )


class CreateScheduledModelExplainabilityMonitorStep(LambdaStepBase):
    def __init__(self, 
        name: str,
        function_name: str,
        execution_role_arn: str,
        deploy_type:str,
        monitor_dir:str,
        monitor_name:str,
        endpoint_name:str,
        data_cature_dir:str,
        image_uri:str = "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
        instance_count:int = 1,
        instance_type:str = 'ml.m5.large',
        volume_size_in_gb:str = 20,
        max_runtime_in_seconds:str = 1800,
        dataset_format:str = {'Csv': {'Header': True}}, # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}} 
        schedule_expression:str = 'cron(0 * ? * * *)',
        data_analysis_start_time:str = "-PT2H",
        data_analysis_end_time:str = "-PT1H",
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):

        inputs={
            'name': monitor_name,
            'role_arn': execution_role_arn,
            'deploy_type': deploy_type,
            'monitor_dir': monitor_dir,
            'image_uri': image_uri,
            'instance_count': instance_count,
            'instance_type': instance_type,
            'volume_size_in_gb': volume_size_in_gb,
            'max_runtime_in_seconds': max_runtime_in_seconds,
            'dataset_format': dataset_format,
            'schedule_expression': schedule_expression,
            'data_analysis_start_time': data_analysis_start_time,
            'data_analysis_end_time': data_analysis_end_time,
            'endpoint_name': endpoint_name,
            'data_cature_dir': data_cature_dir,
        }

        outputs=[]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/schedule_monitors.py',
            handler='schedule_monitors.model_explainability_handler',
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )


class CreateScheduledModelQualityMonitorStep(LambdaStepBase):
    def __init__(self, 
        name: str,
        function_name: str,
        execution_role_arn: str,
        deploy_type:str,
        problem_type:str,
        ground_truth_label:str,
        monitor_dir:str,
        ground_truth_dir:str,
        monitor_name:str,
        endpoint_name:str,
        data_cature_dir:str,
        image_uri:str = "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
        instance_count:int = 1,
        instance_type:str = 'ml.m5.large',
        volume_size_in_gb:str = 20,
        max_runtime_in_seconds:str = 1800,
        dataset_format:str = {'Csv': {'Header': True}}, # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}} 
        schedule_expression:str = 'cron(0 * ? * * *)',
        data_analysis_start_time:str = "-PT2H",
        data_analysis_end_time:str = "-PT1H",
        sagemaker_session: Session = None,
        lambda_kwargs:Dict=None,
        **step_kwargs
    ):
        inputs={
            'name': monitor_name,
            'role_arn': execution_role_arn,
            'deploy_type': deploy_type,
            'problem_type': problem_type,
            'ground_truth_label': ground_truth_label,
            'monitor_dir': monitor_dir,
            'ground_truth_dir': ground_truth_dir,
            'image_uri': image_uri,
            'instance_count': instance_count,
            'instance_type': instance_type,
            'volume_size_in_gb': volume_size_in_gb,
            'max_runtime_in_seconds': max_runtime_in_seconds,
            'dataset_format': dataset_format,
            'schedule_expression': schedule_expression,
            'data_analysis_start_time': data_analysis_start_time,
            'data_analysis_end_time': data_analysis_end_time,
            'endpoint_name': endpoint_name,
            'data_cature_dir': data_cature_dir,
        }

        outputs=[]
    
        super().__init__(
            name=name,
            function_name=function_name,
            execution_role_arn=execution_role_arn,
            script='scripts/schedule_monitors.py',
            handler='schedule_monitors.model_quality_handler',
            inputs=inputs,
            outputs=outputs,
            sagemaker_session=sagemaker_session,
            lambda_kwargs=lambda_kwargs,
            **step_kwargs
        )