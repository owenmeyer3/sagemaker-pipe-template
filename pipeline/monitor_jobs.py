

vpc_config={'SecurityGroupIds': ['subnet-001be661bcef4b615','subnet-003ad32933ca43e74'],'Subnets': ['sg-63ef435d']}

##############################################
############### JOB DEFINITIONS ##############
##############################################
def create_data_quality_job_definition(        
    sm_client, 
    role, 
    name, 
    deploy_type, 
    monitor_dir, 
    vpc_config=None,
    endpoint_name=None, 
    data_cature_dir=None, 
    instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800,  
    dataset_format={'Csv': {'Header': True}} # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}}
    ):
    if deploy_type == 'realtime':
        job_input={
            'EndpointInput': {
                'EndpointName': endpoint_name,
                'LocalPath': '/opt/ml/processing/input/endpoint'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }
    else:
        job_input={
            'BatchTransformInput': {
                'DataCapturedDestinationS3Uri': f'{data_cature_dir}/',
                'DatasetFormat': dataset_format,
                'LocalPath': '/opt/ml/processing/input'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }

    response = sm_client.create_data_quality_job_definition(
        JobDefinitionName=name,
        DataQualityBaselineConfig={
            #'BaseliningJobName': 'string',
            "ConstraintsResource": {"S3Uri": f'{monitor_dir}/constraints.json'},
            "StatisticsResource": {"S3Uri": f'{monitor_dir}/statistics.json'}
        },
        DataQualityAppSpecification={
            'ImageUri': "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer"#,
            # 'ContainerEntrypoint': ['string',],
            # 'ContainerArguments': ['string',],
            # 'RecordPreprocessorSourceUri': 'string',
            # 'PostAnalyticsProcessorSourceUri': 'string',
            # 'Environment': {'string': 'string'}
        },
        DataQualityJobInput=job_input,
        DataQualityJobOutputConfig={
            'MonitoringOutputs': [
                {
                    'S3Output': {
                        'S3Uri': f'{monitor_dir}/reports',
                        'LocalPath': '/opt/ml/processing/output'#,
                        # 'S3UploadMode': 'Continuous'|'EndOfJob'
                    }
                },
            ],
            # 'KmsKeyId': 'string'
        },
        JobResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': instance_type,
                'VolumeSizeInGB': volume_size_in_gb#,
                # 'VolumeKmsKeyId': 'string'
            }
        },
        NetworkConfig={
            # 'EnableInterContainerTrafficEncryption': True|False,
            # 'EnableNetworkIsolation': True|False,
            'VpcConfig': vpc_config
        },
        RoleArn=role,
        StoppingCondition={
            'MaxRuntimeInSeconds': max_runtime_in_seconds
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )

    return response


def create_model_bias_job_definition(        
    sm_client, 
    role, 
    name, 
    deploy_type, 
    monitor_dir, 
    ground_truth_dir,
    vpc_config=None,
    endpoint_name=None, 
    data_cature_dir=None, 
    instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800,  
    dataset_format={'Csv': {'Header': True}} # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}}
    ):
    if deploy_type == 'realtime':
        job_input={
            'EndpointInput': {
                'EndpointName': endpoint_name,
                'LocalPath': '/opt/ml/processing/input/endpoint'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }
    else:
        job_input={
            'BatchTransformInput': {
                'DataCapturedDestinationS3Uri': f'{data_cature_dir}/',
                'DatasetFormat': dataset_format,
                'LocalPath': '/opt/ml/processing/input'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }
    job_input['GroundTruthS3Input']={'S3Uri': ground_truth_dir}

    response = sm_client.create_model_bias_job_definition(
        JobDefinitionName=name,
        ModelBiasBaselineConfig={
            #'BaseliningJobName': 'string',
            "ConstraintsResource": {"S3Uri": f'{monitor_dir}/constraints.json'}
        },
        ModelBiasAppSpecification={
            'ImageUri': "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
            'ConfigUri': 'string',
            # 'Environment': {'string': 'string'}
        },
        ModelBiasJobInput=job_input,
        ModelBiasJobOutputConfig={
            'MonitoringOutputs': [
                {
                    'S3Output': {
                        'S3Uri': f'{monitor_dir}/reports',
                        'LocalPath': '/opt/ml/processing/output'#,
                        # 'S3UploadMode': 'Continuous'|'EndOfJob'
                    }
                },
            ],
            # 'KmsKeyId': 'string'
        },
        JobResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': instance_type,
                'VolumeSizeInGB': volume_size_in_gb#,
                # 'VolumeKmsKeyId': 'string'
            }
        },
        NetworkConfig={
            # 'EnableInterContainerTrafficEncryption': True|False,
            # 'EnableNetworkIsolation': True|False,
            'VpcConfig': vpc_config
        },
        RoleArn=role,
        StoppingCondition={
            'MaxRuntimeInSeconds': max_runtime_in_seconds
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )

    return response



def create_model_explainability_job_definition(        
    sm_client, 
    role, 
    name, 
    deploy_type, 
    monitor_dir,
    ground_truth_dir,
    vpc_config=None,
    endpoint_name=None, 
    data_cature_dir=None, 
    instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800,  
    dataset_format={'Csv': {'Header': True}} # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}}
    ):
    if deploy_type == 'realtime':
        job_input={
            'EndpointInput': {
                'EndpointName': endpoint_name,
                'LocalPath': '/opt/ml/processing/input/endpoint'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }
    else:
        job_input={
            'BatchTransformInput': {
                'DataCapturedDestinationS3Uri': f'{data_cature_dir}/',
                'DatasetFormat': dataset_format,
                'LocalPath': '/opt/ml/processing/input'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }
    job_input['GroundTruthS3Input']={'S3Uri': ground_truth_dir}

    response = sm_client.create_model_explainability_job_definition(
        JobDefinitionName=name,
        ModelExplainabilityBaselineConfig={
            #'BaseliningJobName': 'string',
            "ConstraintsResource": {"S3Uri": f'{monitor_dir}/constraints.json'}
        },
        ModelExplainabilityAppSpecification={
            'ImageUri': "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
            'ConfigUri': 'string',
            # 'Environment': {'string': 'string'}
        },
        ModelExplainabilityJobInput=job_input,
        ModelExplainabilityJobOutputConfig={
            'MonitoringOutputs': [
                {
                    'S3Output': {
                        'S3Uri': f'{monitor_dir}/reports',
                        'LocalPath': '/opt/ml/processing/output'#,
                        # 'S3UploadMode': 'Continuous'|'EndOfJob'
                    }
                },
            ],
            # 'KmsKeyId': 'string'
        },
        JobResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': instance_type,
                'VolumeSizeInGB': volume_size_in_gb#,
                # 'VolumeKmsKeyId': 'string'
            }
        },
        NetworkConfig={
            # 'EnableInterContainerTrafficEncryption': True|False,
            # 'EnableNetworkIsolation': True|False,
            'VpcConfig': vpc_config
        },
        RoleArn=role,
        StoppingCondition={
            'MaxRuntimeInSeconds': max_runtime_in_seconds
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )

    return response



def create_model_quality_job_definition(        
    sm_client, 
    role, 
    name, 
    deploy_type, 
    monitor_dir, 
    vpc_config=None,
    endpoint_name=None, 
    data_cature_dir=None, 
    instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800,  
    dataset_format={'Csv': {'Header': True}} # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}}
):
    if deploy_type == 'realtime':
        job_input={
            'EndpointInput': {
                'EndpointName': endpoint_name,
                'LocalPath': '/opt/ml/processing/input/endpoint'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }
    else:
        job_input={
            'BatchTransformInput': {
                'DataCapturedDestinationS3Uri': f'{data_cature_dir}/',
                'DatasetFormat': dataset_format,
                'LocalPath': '/opt/ml/processing/input'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }

    response = sm_client.create_model_quality_job_definition(
        JobDefinitionName=name,
        ModelQualityBaselineConfig={
            #'BaseliningJobName': 'string',
            "ConstraintsResource": {"S3Uri": f'{monitor_dir}/constraints.json'}
        },
        ModelQualityAppSpecification={
            'ImageUri': "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
            # 'ContainerEntrypoint': ['string',],
            # 'ContainerArguments': ['string',],
            # 'RecordPreprocessorSourceUri': 'string',
            # 'PostAnalyticsProcessorSourceUri': 'string',
            # 'Environment': {'string': 'string'}
        },
        ModelQualityJobInput=job_input,
        ModelQualityJobOutputConfig={
            'MonitoringOutputs': [
                {
                    'S3Output': {
                        'S3Uri': f'{monitor_dir}/reports',
                        'LocalPath': '/opt/ml/processing/output'#,
                        # 'S3UploadMode': 'Continuous'|'EndOfJob'
                    }
                },
            ],
            # 'KmsKeyId': 'string'
        },
        JobResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': instance_type,
                'VolumeSizeInGB': volume_size_in_gb#,
                # 'VolumeKmsKeyId': 'string'
            }
        },
        NetworkConfig={
            # 'EnableInterContainerTrafficEncryption': True|False,
            # 'EnableNetworkIsolation': True|False,
            'VpcConfig': vpc_config
        },
        RoleArn=role,
        StoppingCondition={
            'MaxRuntimeInSeconds': max_runtime_in_seconds
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )
    return response


##############################################
############### CREATE SCHEDULES ##############
##############################################
# 'DataQuality'|'ModelQuality'|'ModelBias'|'ModelExplainability'
def create_data_quality_monitoring_schedule(
    sm_client, 
    name,
    role,
    deploy_type,
    monitor_dir,
    schedule_expression, 
    data_analysis_start_time, 
    data_analysis_end_time,
    vpc_config={'SecurityGroupIds': ['subnet-001be661bcef4b615','subnet-003ad32933ca43e74'],'Subnets': ['sg-63ef435d']}
):

    job_definition_name = f'{name}-job'

    response = create_data_quality_job_definition(        
        sm_client, 
        role, 
        job_definition_name, 
        deploy_type, 
        monitor_dir, 
        vpc_config=vpc_config,
        endpoint_name=None, 
        data_cature_dir=None, 
        instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800,  
        dataset_format={'Csv': {'Header': True}} # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}}
    )

    response = sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            'ScheduleConfig': {
                'ScheduleExpression': schedule_expression,
                'DataAnalysisStartTime': data_analysis_start_time,
                'DataAnalysisEndTime': data_analysis_end_time
            },
            'MonitoringJobDefinitionName': job_definition_name,
            'MonitoringType': 'DataQuality'
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )
    return response


def create_model_bias_monitoring_schedule(
    sm_client, 
    name,
    role,
    deploy_type,
    monitor_dir,
    ground_truth_dir,
    schedule_expression, 
    data_analysis_start_time, 
    data_analysis_end_time,
    vpc_config={'SecurityGroupIds': ['subnet-001be661bcef4b615','subnet-003ad32933ca43e74'],'Subnets': ['sg-63ef435d']}
):

    job_definition_name = f'{name}-job'

    response = create_model_bias_job_definition(        
        sm_client, 
        role, 
        name, 
        deploy_type, 
        monitor_dir, 
        ground_truth_dir,
        vpc_config=vpc_config,
        endpoint_name=None, 
        data_cature_dir=None, 
        instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800,  
        dataset_format={'Csv': {'Header': True}} # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}}
    )

    response = sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            'ScheduleConfig': {
                'ScheduleExpression': schedule_expression,
                'DataAnalysisStartTime': data_analysis_start_time,
                'DataAnalysisEndTime': data_analysis_end_time
            },
            'MonitoringJobDefinitionName': job_definition_name,
            'MonitoringType': 'ModelBias'
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )
    return response


def create_model_explainability_monitoring_schedule(
    sm_client, 
    name,
    role,
    deploy_type,
    monitor_dir,
    ground_truth_dir,
    schedule_expression, 
    data_analysis_start_time, 
    data_analysis_end_time,
    vpc_config={'SecurityGroupIds': ['subnet-001be661bcef4b615','subnet-003ad32933ca43e74'],'Subnets': ['sg-63ef435d']}
):

    job_definition_name = f'{name}-job'

    response = create_model_explainability_job_definition(        
        sm_client, 
        role, 
        name, 
        deploy_type, 
        monitor_dir, 
        ground_truth_dir,
        vpc_config=vpc_config,
        endpoint_name=None, 
        data_cature_dir=None, 
        instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800,  
        dataset_format={'Csv': {'Header': True}} # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}}
    )

    response = sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            'ScheduleConfig': {
                'ScheduleExpression': schedule_expression,
                'DataAnalysisStartTime': data_analysis_start_time,
                'DataAnalysisEndTime': data_analysis_end_time
            },
            'MonitoringJobDefinitionName': job_definition_name,
            'MonitoringType': 'ModelExplainability'
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )
    return response

def create_model_quality_monitoring_schedule(
    sm_client, 
    name,
    role,
    deploy_type,
    monitor_dir,
    schedule_expression, 
    data_analysis_start_time, 
    data_analysis_end_time,
    vpc_config={'SecurityGroupIds': ['subnet-001be661bcef4b615','subnet-003ad32933ca43e74'],'Subnets': ['sg-63ef435d']}
):

    job_definition_name = f'{name}-job'

    response = create_model_quality_job_definition(        
        sm_client, 
        role, 
        name, 
        deploy_type, 
        monitor_dir, 
        vpc_config=vpc_config,
        endpoint_name=None, 
        data_cature_dir=None, 
        instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800,  
        dataset_format={'Csv': {'Header': True}} # {'Csv':{'Header': True|False},'Json': {'Line': True|False}, Parquet': {}}
    )

    response = sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            'ScheduleConfig': {
                'ScheduleExpression': schedule_expression,
                'DataAnalysisStartTime': data_analysis_start_time,
                'DataAnalysisEndTime': data_analysis_end_time
            },
            'MonitoringJobDefinitionName': job_definition_name,
            'MonitoringType': 'ModelQuality'
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )
    return response


