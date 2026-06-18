from monitor_inputs import get_monitoring_job_input


def create_scheduled_data_quality_monitor(
        sm_client, 
        role, 
        name, 
        deploy_type, 
        schedule_expression, 
        monitor_dir, 
        endpoint_name=None, 
        data_analysis_start_time="-PT1H", data_analysis_end_time="-PT0H", 
        data_cature_dir=None, 
        instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800, 
        vpc_config={'SecurityGroupIds': ['subnet-001be661bcef4b615','subnet-003ad32933ca43e74'], 'Subnets': ['sg-63ef435d']}, 
        dataset_format={'Csv': {'Header': True|False}}
    ):

    job_input=get_monitoring_job_input(deploy_type, endpoint_name=endpoint_name, data_cature_dir=data_cature_dir, dataset_format=dataset_format)

    return sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            'ScheduleConfig': {
                'ScheduleExpression': schedule_expression,
                'DataAnalysisStartTime': data_analysis_start_time,
                'DataAnalysisEndTime': data_analysis_end_time
            },
            'MonitoringJobDefinition': {
                'BaselineConfig': {
                    #'BaseliningJobName': 'string',
                    "ConstraintsResource": {"S3Uri": f'{monitor_dir}/constraints.json'},
                    "StatisticsResource": {"S3Uri": f'{monitor_dir}/statistics.json'}
                },
                'MonitoringInputs': [job_input],
                'MonitoringOutputConfig': {
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
                'MonitoringResources': {
                    'ClusterConfig': {
                        'InstanceCount': 1,
                        'InstanceType': instance_type,
                        'VolumeSizeInGB': volume_size_in_gb#,
                        # 'VolumeKmsKeyId': 'string'
                    }
                },
                'MonitoringAppSpecification': {
                    'ImageUri': "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer"#,
                    # 'ContainerEntrypoint': ['string',],
                    # 'ContainerArguments': ['string',],
                    # 'RecordPreprocessorSourceUri': 'string',
                    # 'PostAnalyticsProcessorSourceUri': 'string',
                    # 'Environment': {'string': 'string'}
                },
                'StoppingCondition': {
                    'MaxRuntimeInSeconds': max_runtime_in_seconds
                },
                'Environment': {'string': 'string'},
                'NetworkConfig': {
                    # 'EnableInterContainerTrafficEncryption': True|False,
                    # 'EnableNetworkIsolation': True|False,
                    'VpcConfig': vpc_config
                },
                'RoleArn': role
            },
            # 'MonitoringJobDefinitionName': f'{name}-job', -- dont include since were making it here
            'MonitoringType': 'DataQuality' # |'ModelQuality'|'ModelBias'|'ModelExplainability'
        },
        # Tags=[{'Key': 'string', 'Value': 'string'},]
    )




# def create_scheduled_data_quality_monitor(sm_client, role, name, deploy_type, schedule_expression, constraints_file, statistics_file, monitor_dir, endpoint_name=None, data_cature_dir=None, instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800):

#     if deploy_type == 'realtime':
#         job_input={
#             'EndpointInput': {
#                 'EndpointName': endpoint_name,
#                 'LocalPath': '/opt/ml/processing/input/endpoint',
#                 # 'S3InputMode': 'Pipe'|'File',
#                 # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
#                 # 'FeaturesAttribute': 'string',
#                 # 'InferenceAttribute': 'string',
#                 # 'ProbabilityAttribute': 'string',
#                 # 'ProbabilityThresholdAttribute': 123.0,
#                 # 'StartTimeOffset': 'string',
#                 # 'EndTimeOffset': 'string',
#                 # 'ExcludeFeaturesAttribute': 'string'
#             }
#         }
#     else:
#         job_input={
#             'BatchTransformInput': {
#                 'DataCapturedDestinationS3Uri': f'{data_cature_dir}/',
#                 'DatasetFormat': {
#                     'Csv': {'Header': True|False},
#                     # 'Json': {'Line': True|False},
#                     # 'Parquet': {}
#                 },
#                 'LocalPath': '/opt/ml/processing/input',
#                 # 'S3InputMode': 'Pipe'|'File',
#                 # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
#                 # 'FeaturesAttribute': 'string',
#                 # 'InferenceAttribute': 'string',
#                 # 'ProbabilityAttribute': 'string',
#                 # 'ProbabilityThresholdAttribute': 123.0,
#                 # 'StartTimeOffset': 'string',
#                 # 'EndTimeOffset': 'string',
#                 # 'ExcludeFeaturesAttribute': 'string'
#             }
#         }

#     response = sm_client.create_data_quality_job_definition(
#         JobDefinitionName=f'{name}-job',
#         DataQualityBaselineConfig={
#             #'BaseliningJobName': 'string',
#             "ConstraintsResource": {"S3Uri": constraints_file},
#             "StatisticsResource": {"S3Uri": statistics_file}
#         },
#         DataQualityAppSpecification={
#             'ImageUri': "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
#             # 'ContainerEntrypoint': ['string',],
#             # 'ContainerArguments': ['string',],
#             # 'RecordPreprocessorSourceUri': 'string',
#             # 'PostAnalyticsProcessorSourceUri': 'string',
#             # 'Environment': {'string': 'string'}
#         },
#         DataQualityJobInput=job_input,
#         DataQualityJobOutputConfig={
#             'MonitoringOutputs': [
#                 {
#                     'S3Output': {
#                         'S3Uri': f'{monitor_dir}/reports',
#                         'LocalPath': '/opt/ml/processing/output',
#                         # 'S3UploadMode': 'Continuous'|'EndOfJob'
#                     }
#                 },
#             ],
#             # 'KmsKeyId': 'string'
#         },
#         JobResources={
#             'ClusterConfig': {
#                 'InstanceCount': 1,
#                 'InstanceType': instance_type,
#                 'VolumeSizeInGB': volume_size_in_gb,
#                 # 'VolumeKmsKeyId': 'string'
#             }
#         },
#         NetworkConfig={
#             # 'EnableInterContainerTrafficEncryption': True|False,
#             # 'EnableNetworkIsolation': True|False,
#             'VpcConfig': {
#                 'SecurityGroupIds': ['string',],
#                 'Subnets': ['string',]
#             }
#         },
#         RoleArn=role,
#         StoppingCondition={
#             'MaxRuntimeInSeconds': max_runtime_in_seconds
#         },
#         # Tags=[{'Key': 'string', 'Value': 'string'},]
#     )


#     response = sm_client.create_monitoring_schedule(
#         MonitoringScheduleName=name,
#         MonitoringScheduleConfig={
#             'ScheduleConfig': {
#                 'ScheduleExpression': schedule_expression,
#                 'DataAnalysisStartTime': "-PT1H",
#                 'DataAnalysisEndTime': "-PT0H"
#             },
#             'MonitoringJobDefinition': {
#                 'BaselineConfig': {
#                     'BaseliningJobName': 'string',
#                     'ConstraintsResource': {
#                         'S3Uri': 'string'
#                     },
#                     'StatisticsResource': {
#                         'S3Uri': 'string'
#                     }
#                 },
#                 'MonitoringInputs': [
#                     {
#                         'EndpointInput': {
#                             'EndpointName': 'string',
#                             'LocalPath': 'string',
#                             'S3InputMode': 'Pipe'|'File',
#                             'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
#                             'FeaturesAttribute': 'string',
#                             'InferenceAttribute': 'string',
#                             'ProbabilityAttribute': 'string',
#                             'ProbabilityThresholdAttribute': 123.0,
#                             'StartTimeOffset': 'string',
#                             'EndTimeOffset': 'string',
#                             'ExcludeFeaturesAttribute': 'string'
#                         },
#                         'BatchTransformInput': {
#                             'DataCapturedDestinationS3Uri': 'string',
#                             'DatasetFormat': {
#                                 'Csv': {
#                                     'Header': True|False
#                                 },
#                                 'Json': {
#                                     'Line': True|False
#                                 },
#                                 'Parquet': {}

#                             },
#                             'LocalPath': 'string',
#                             'S3InputMode': 'Pipe'|'File',
#                             'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
#                             'FeaturesAttribute': 'string',
#                             'InferenceAttribute': 'string',
#                             'ProbabilityAttribute': 'string',
#                             'ProbabilityThresholdAttribute': 123.0,
#                             'StartTimeOffset': 'string',
#                             'EndTimeOffset': 'string',
#                             'ExcludeFeaturesAttribute': 'string'
#                         }
#                     },
#                 ],
#                 'MonitoringOutputConfig': {
#                     'MonitoringOutputs': [
#                         {
#                             'S3Output': {
#                                 'S3Uri': 'string',
#                                 'LocalPath': 'string',
#                                 'S3UploadMode': 'Continuous'|'EndOfJob'
#                             }
#                         },
#                     ],
#                     'KmsKeyId': 'string'
#                 },
#                 'MonitoringResources': {
#                     'ClusterConfig': {
#                         'InstanceCount': 123,
#                         'InstanceType': 'ml.t3.medium'|'ml.t3.large'|'ml.t3.xlarge'|'ml.t3.2xlarge'|'ml.m4.xlarge'|'ml.m4.2xlarge'|'ml.m4.4xlarge'|'ml.m4.10xlarge'|'ml.m4.16xlarge'|'ml.c4.xlarge'|'ml.c4.2xlarge'|'ml.c4.4xlarge'|'ml.c4.8xlarge'|'ml.p2.xlarge'|'ml.p2.8xlarge'|'ml.p2.16xlarge'|'ml.p3.2xlarge'|'ml.p3.8xlarge'|'ml.p3.16xlarge'|'ml.c5.xlarge'|'ml.c5.2xlarge'|'ml.c5.4xlarge'|'ml.c5.9xlarge'|'ml.c5.18xlarge'|'ml.m5.large'|'ml.m5.xlarge'|'ml.m5.2xlarge'|'ml.m5.4xlarge'|'ml.m5.12xlarge'|'ml.m5.24xlarge'|'ml.r5.large'|'ml.r5.xlarge'|'ml.r5.2xlarge'|'ml.r5.4xlarge'|'ml.r5.8xlarge'|'ml.r5.12xlarge'|'ml.r5.16xlarge'|'ml.r5.24xlarge'|'ml.g4dn.xlarge'|'ml.g4dn.2xlarge'|'ml.g4dn.4xlarge'|'ml.g4dn.8xlarge'|'ml.g4dn.12xlarge'|'ml.g4dn.16xlarge'|'ml.g5.xlarge'|'ml.g5.2xlarge'|'ml.g5.4xlarge'|'ml.g5.8xlarge'|'ml.g5.16xlarge'|'ml.g5.12xlarge'|'ml.g5.24xlarge'|'ml.g5.48xlarge'|'ml.r5d.large'|'ml.r5d.xlarge'|'ml.r5d.2xlarge'|'ml.r5d.4xlarge'|'ml.r5d.8xlarge'|'ml.r5d.12xlarge'|'ml.r5d.16xlarge'|'ml.r5d.24xlarge'|'ml.g6.xlarge'|'ml.g6.2xlarge'|'ml.g6.4xlarge'|'ml.g6.8xlarge'|'ml.g6.12xlarge'|'ml.g6.16xlarge'|'ml.g6.24xlarge'|'ml.g6.48xlarge'|'ml.g6e.xlarge'|'ml.g6e.2xlarge'|'ml.g6e.4xlarge'|'ml.g6e.8xlarge'|'ml.g6e.12xlarge'|'ml.g6e.16xlarge'|'ml.g6e.24xlarge'|'ml.g6e.48xlarge'|'ml.m6i.large'|'ml.m6i.xlarge'|'ml.m6i.2xlarge'|'ml.m6i.4xlarge'|'ml.m6i.8xlarge'|'ml.m6i.12xlarge'|'ml.m6i.16xlarge'|'ml.m6i.24xlarge'|'ml.m6i.32xlarge'|'ml.c6i.xlarge'|'ml.c6i.2xlarge'|'ml.c6i.4xlarge'|'ml.c6i.8xlarge'|'ml.c6i.12xlarge'|'ml.c6i.16xlarge'|'ml.c6i.24xlarge'|'ml.c6i.32xlarge'|'ml.m7i.large'|'ml.m7i.xlarge'|'ml.m7i.2xlarge'|'ml.m7i.4xlarge'|'ml.m7i.8xlarge'|'ml.m7i.12xlarge'|'ml.m7i.16xlarge'|'ml.m7i.24xlarge'|'ml.m7i.48xlarge'|'ml.c7i.large'|'ml.c7i.xlarge'|'ml.c7i.2xlarge'|'ml.c7i.4xlarge'|'ml.c7i.8xlarge'|'ml.c7i.12xlarge'|'ml.c7i.16xlarge'|'ml.c7i.24xlarge'|'ml.c7i.48xlarge'|'ml.r7i.large'|'ml.r7i.xlarge'|'ml.r7i.2xlarge'|'ml.r7i.4xlarge'|'ml.r7i.8xlarge'|'ml.r7i.12xlarge'|'ml.r7i.16xlarge'|'ml.r7i.24xlarge'|'ml.r7i.48xlarge'|'ml.p5.4xlarge'|'ml.g7e.2xlarge'|'ml.g7e.4xlarge'|'ml.g7e.8xlarge'|'ml.g7e.12xlarge'|'ml.g7e.24xlarge'|'ml.g7e.48xlarge',
#                         'VolumeSizeInGB': 123,
#                         'VolumeKmsKeyId': 'string'
#                     }
#                 },
#                 'MonitoringAppSpecification': {
#                     'ImageUri': 'string',
#                     'ContainerEntrypoint': [
#                         'string',
#                     ],
#                     'ContainerArguments': [
#                         'string',
#                     ],
#                     'RecordPreprocessorSourceUri': 'string',
#                     'PostAnalyticsProcessorSourceUri': 'string'
#                 },
#                 'StoppingCondition': {
#                     'MaxRuntimeInSeconds': 123
#                 },
#                 'Environment': {
#                     'string': 'string'
#                 },
#                 'NetworkConfig': {
#                     'EnableInterContainerTrafficEncryption': True|False,
#                     'EnableNetworkIsolation': True|False,
#                     'VpcConfig': {
#                         'SecurityGroupIds': [
#                             'string',
#                         ],
#                         'Subnets': [
#                             'string',
#                         ]
#                     }
#                 },
#                 'RoleArn': 'string'
#             },
#             'MonitoringJobDefinitionName': 'string',
#             'MonitoringType': 'DataQuality'|'ModelQuality'|'ModelBias'|'ModelExplainability'
#         },
#         # Tags=[{'Key': 'string', 'Value': 'string'},]
#     )



#     if deploy_type == 'realtime':
#         monitoring_inputs=[{
#             "EndpointInput": {
#                 "EndpointName": endpoint_name,
#                 "LocalPath": "/opt/ml/processing/input/endpoint"
#             }
#         }]
#     else:
#         monitoring_inputs=[{
#             "BatchTransformInput": {
#                 "DataCapturedDestinationS3Uri": f"{data_cature_dir}/",
#                 "LocalPath": "/opt/ml/processing/input",
#                 "DatasetFormat": {
#                     "Csv": {"Header": False}
#                 }
#             }
#         }]

#     sm_client.create_monitoring_schedule(
#         MonitoringScheduleName=name,
#             MonitoringScheduleConfig={
#             "ScheduleConfig": {
#                 "ScheduleExpression": schedule_expression# "cron(0 * ? * * *)"
#             },
#             "MonitoringJobDefinition": {
#                 "BaselineConfig": {
#                     "ConstraintsResource": {"S3Uri": constraints_file},
#                     "StatisticsResource": {"S3Uri": statistics_file}
#                 },
#                 "MonitoringInputs": monitoring_inputs,
#                 "MonitoringOutputConfig": {
#                     "MonitoringOutputs": [{
#                         "S3Output": {
#                             "S3Uri": f'{monitor_dir}/reports',
#                             "LocalPath": "/opt/ml/processing/output"
#                         }
#                     }]
#                 },
#                 "MonitoringResources": {
#                     "ClusterConfig": {
#                         "InstanceCount": 1,
#                         "InstanceType": instance_type,
#                         "VolumeSizeInGB": volume_size_in_gb
#                     }
#                 },
#                 "MonitoringAppSpecification": {
#                     "ImageUri": "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer"
#                 },
#                 "RoleArn": role,
#                 "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_in_seconds}
#             },
#             "MonitoringType": "DataQuality"
#         }
#     )


def create_scheduled_model_quality_monitor(sm_client, role, name, deploy_type, schedule_expression, constraints_file, ground_truth_input, monitor_dir, endpoint_name=None, data_cature_dir=None, instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800):
    
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
                },
                "InferenceAttribute": "0",
                "StartTimeOffset": "-PT1H",
                "EndTimeOffset": "-PT0H"
            }
        }]
    
    sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            "ScheduleConfig": {
                "ScheduleExpression": schedule_expression
            },
            "MonitoringJobDefinition": {
                "BaselineConfig": {
                    "ConstraintsResource": {"S3Uri": constraints_file},
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
                    "ImageUri": "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
                    "ProblemType": "Regression"
                },
                "RoleArn": role,
                "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_in_seconds},
                "Environment": {
                    "ground_truth_input": ground_truth_input
                }
            },
            "MonitoringType": "ModelQuality"
        }
    )

    
def create_scheduled_data_bias_monitor(sm_client, role, name, deploy_type, schedule_expression, constraints_file, analysis_config_file, monitor_dir, endpoint_name=None, data_cature_dir=None, instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800):
    
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
                "DatasetFormat": {"Csv": {"Header": False}}
            }
        }]
    
    sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            "ScheduleConfig": {
                "ScheduleExpression": schedule_expression
            },
            "MonitoringJobDefinition": {
                "BaselineConfig": {
                    "ConstraintsResource": {"S3Uri": constraints_file}
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
                    "ImageUri": "205585389593.dkr.ecr.us-east-1.amazonaws.com/sagemaker-clarify-processing:1.0",
                    "ConfigUri": analysis_config_file
                },
                "RoleArn": role,
                "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_in_seconds}
            },
            "MonitoringType": "DataBias"
        }
    )


def create_scheduled_model_bias_monitor(sm_client, role, name, deploy_type, schedule_expression, constraints_file, analysis_config_file, monitor_dir, endpoint_name=None, data_cature_dir=None, instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800):

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
                "DatasetFormat": {"Csv": {"Header": False}},
                "InferenceAttribute": "0"
            }
        }]

    sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            "ScheduleConfig": {
                "ScheduleExpression": schedule_expression
            },
            "MonitoringJobDefinition": {
                "BaselineConfig": {
                    "ConstraintsResource": {"S3Uri": constraints_file}
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
                    "ImageUri": "205585389593.dkr.ecr.us-east-1.amazonaws.com/sagemaker-clarify-processing:1.0",
                    "ConfigUri": analysis_config_file
                },
                "RoleArn": role,
                "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_in_seconds}
            },
            "MonitoringType": "ModelBias"
        }
    )


def create_scheduled_model_explainability_monitor(sm_client, role, name, deploy_type, schedule_expression, constraints_file, analysis_config_file, monitor_dir, endpoint_name=None, data_cature_dir=None, instance_type='ml.m5.large', volume_size_in_gb=20, max_runtime_in_seconds=1800):
    
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
                "DatasetFormat": {"Csv": {"Header": False}},
                "InferenceAttribute": "0"
            }
        }]
    
    sm_client.create_monitoring_schedule(
        MonitoringScheduleName=name,
        MonitoringScheduleConfig={
            "ScheduleConfig": {
                "ScheduleExpression": schedule_expression
            },
            "MonitoringJobDefinition": {
                "BaselineConfig": {
                    "ConstraintsResource": {"S3Uri": constraints_file}
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
                    "ImageUri": "205585389593.dkr.ecr.us-east-1.amazonaws.com/sagemaker-clarify-processing:1.0",
                    "ConfigUri": analysis_config_file
                },
                "RoleArn": role,
                "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_in_seconds}
            },
            "MonitoringType": "ModelExplainability"
        }
    )