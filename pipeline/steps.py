from sagemaker.core.lambda_helper import Lambda
from sagemaker.mlops.workflow.lambda_step import LambdaOutputTypeEnum, LambdaStep, LambdaOutput


class GetOrCreateModelFromRegistryStep(LambdaStep):
    def __init__(self, 
        name, 
        execution_role_arn, 
        model_package_group_name_param, 
        model_package_version_param, 
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func=Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/get_or_create_model_from_registry.py',  # path to your file
            handler='get_or_create_model_from_registry.handler',    # filename.function_name
            timeout=timeout,
            memory_size=memory_size
        )

        inputs={
            'model_package_group_name': model_package_group_name_param,
            'model_package_version': model_package_version_param
        }

        outputs=[
            LambdaOutput(output_name='model_name', output_type=LambdaOutputTypeEnum.String),
            LambdaOutput(output_name='model_package_arn', output_type=LambdaOutputTypeEnum.String)
        ]

        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)


class DeployEndpointStep(LambdaStep):
    def __init__(self, 
        name, 
        execution_role_arn, 
        model_name_param, 
        model_package_group_name_param, 
        model_package_version_param, 
        endpoint_instance_type,
        data_capture_dir,
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func = Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/deploy_endpoint.py',
            handler='deploy_endpoint.handler',
            timeout=timeout,  # endpoints take time to deploy
            memory_size=memory_size
        )

        inputs={
            'model_name': model_name_param,
            'model_package_group_name_param': model_package_group_name_param,
            'model_package_version_param': model_package_version_param,
            'instance_type': endpoint_instance_type,
            'data_capture_path':data_capture_dir
        }

        outputs=[
            LambdaOutput(output_name='endpoint_name', output_type=LambdaOutputTypeEnum.String)
        ]
        
        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)


class PrepBaselineSetsStep(LambdaStep):
    def __init__(self, 
        name, 
        execution_role_arn, 
        baseline_file,
        target_name,
        target_type,
        baseline_X_file_dest_dir,
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func = Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/baselining.py',
            handler='baselining.prep_baseline_sets_handler',
            timeout=timeout,
            memory_size=memory_size
        )

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
        
        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)


class GetBaselinePredsStep(LambdaStep):
    def __init__(self, 
        name, 
        execution_role_arn, 
        transform_out_dir,
        baseline_X_filename,
        baseline_pred_file_dest,
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func = Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/baselining.py',
            handler='baselining.get_baseline_preds_handler',
            timeout=timeout,
            memory_size=memory_size
        )

        inputs={
            'transform_out_dir': transform_out_dir,
            'baseline_X_filename':baseline_X_filename,
            'baseline_pred_file_dest':baseline_pred_file_dest
        }

        outputs=[
            LambdaOutput(output_name='baseline_pred_file', output_type=LambdaOutputTypeEnum.String),
        ]
        
        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)


class MakeBaselineSetsStep(LambdaStep):
    def __init__(self, 
        name, 
        execution_role_arn, 
        target_name,
        prediction_name,
        target_type,
        baseline_file,
        baseline_pred_file,
        dq_monitor_dir,
        db_monitor_dir,
        mq_monitor_dir,
        mb_monitor_dir,
        me_monitor_dir,
        train_file,
        train_X_file,
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func = Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/baselining.py',
            handler='baselining.make_baseline_sets_handler',
            timeout=timeout,
            memory_size=memory_size
        )

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

        outputs=[
            LambdaOutput(output_name='result', output_type=LambdaOutputTypeEnum.String),
        ]
        
        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)


class CreateScheduledDataQualityMonitorStep(LambdaStep):
    def __init__(self, 
        name,
        execution_role_arn,
        role_param,
        endpoint_name,
        data_cature_dir,
        deploy_type,
        monitor_dir,
        schedule_expression,
        data_analysis_start_time,
        data_analysis_end_time,
        vpc_config={'SecurityGroupIds': ['subnet-001be661bcef4b615','subnet-003ad32933ca43e74'],'Subnets': ['sg-63ef435d']},
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func = Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/schedule_monitors.py',
            handler='schedule_monitors.data_quality_handler',
            timeout=timeout,
            memory_size=memory_size
        )

        inputs={
            'role': role_param,
            'endpoint_name': endpoint_name,
            'data_cature_dir': data_cature_dir,
            'deploy_type': deploy_type,
            'monitor_dir': monitor_dir,
            'schedule_expression': schedule_expression,
            'data_analysis_start_time': data_analysis_start_time,
            'data_analysis_end_time': data_analysis_end_time,
            'vpc_config': vpc_config
        }

        outputs=[
            LambdaOutput(output_name='result', output_type=LambdaOutputTypeEnum.String),
        ]
        
        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)


class CreateScheduledModelBiasMonitorStep(LambdaStep):
    def __init__(self, 
        name, 
        execution_role_arn,
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func = Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/schedule_monitors.py',
            handler='schedule_monitors.model_bias_handler',
            timeout=timeout,
            memory_size=memory_size
        )

        inputs={
        }

        outputs=[
            LambdaOutput(output_name='result', output_type=LambdaOutputTypeEnum.String),
        ]
        
        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)


class CreateScheduledModelExplainabilityMonitorStep(LambdaStep):
    def __init__(self, 
        name, 
        execution_role_arn,
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func = Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/schedule_monitors.py',
            handler='schedule_monitors.model_explainability_handler',
            timeout=timeout,
            memory_size=memory_size
        )

        inputs={
        }

        outputs=[
            LambdaOutput(output_name='result', output_type=LambdaOutputTypeEnum.String),
        ]
        
        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)


class CreateScheduledModelQualityMonitorStep(LambdaStep):
    def __init__(self, 
        name, 
        execution_role_arn,
        depends_on=[], 
        timeout=600, 
        memory_size=128
    ):
        lambda_func = Lambda(
            function_name=name,
            execution_role_arn=execution_role_arn,
            script='scripts/schedule_monitors.py',
            handler='schedule_monitors.model_quality_handler',
            timeout=timeout,
            memory_size=memory_size
        )

        inputs={
        }

        outputs=[
            LambdaOutput(output_name='result', output_type=LambdaOutputTypeEnum.String),
        ]
        
        super().__init__(name=name, lambda_func=lambda_func, inputs=inputs, outputs=outputs, depends_on=depends_on)