import json, boto3

json_definition = """
{
    "Version": "2020-12-01",
    "Metadata": {},
    "Parameters": [
        {
        "Name": "ModelPackageVersion",
        "Type": "String",
        "DefaultValue": "1"
        },
        {
        "Name": "Action",
        "Type": "String",
        "DefaultValue": "deploy",
        "EnumValues": [
            "deploy",
            "inference"
        ]
        },
        {
        "Name": "BaselineFile",
        "Type": "String",
        "DefaultValue": "aaa"
        },
        {
        "Name": "MonitorInstanceType",
        "Type": "String",
        "DefaultValue": "ml.m5.large"
        },
        {
        "Name": "EndpointInstanceType",
        "Type": "String",
        "DefaultValue": "ml.m5.large"
        },
        {
        "Name": "TransformInstanceType",
        "Type": "String",
        "DefaultValue": "ml.m5.large"
        },
        {
        "Name": "FailOnViolation",
        "Type": "String",
        "DefaultValue": "False"
        },
        {
        "Name": "RegisterNewBaseline",
        "Type": "String",
        "DefaultValue": "False"
        },
        {
        "Name": "MonitorScheduleExpression",
        "Type": "String",
        "DefaultValue": "cron(0 * ? * * *)"
        },
        {
        "Name": "EnableDataQualityMonitoring",
        "Type": "String",
        "DefaultValue": "True"
        },
        {
        "Name": "EnableModelBiasMonitoring",
        "Type": "String",
        "DefaultValue": "True"
        },
        {
        "Name": "EnableModelExplainabilityMonitoring",
        "Type": "String",
        "DefaultValue": "True"
        },
        {
        "Name": "EnableModelQualityMonitoring",
        "Type": "String",
        "DefaultValue": "True"
        },
        {
        "Name": "Environment",
        "Type": "String",
        "DefaultValue": "dev",
        "EnumValues": [
            "prd",
            "dev",
            "stg"
        ]
        },
        {
        "Name": "SnsTopicArn",
        "Type": "String",
        "DefaultValue": "aaa"
        },
        {
        "Name": "EnableSnsNotification",
        "Type": "String",
        "DefaultValue": "False"
        },
        {
        "Name": "GroundTruthDir",
        "Type": "String",
        "DefaultValue": "s3://omm-test-bucket/ground-truth/abalone"
        },
        {
        "Name": "BatchInputDir",
        "Type": "String",
        "DefaultValue": "s3://omm-test-bucket/batch_input/abalone"
        }
    ],
    "PipelineExperimentConfig": {
        "ExperimentName": {
        "Get": "Execution.PipelineName"
        },
        "TrialName": {
        "Get": "Execution.PipelineExecutionId"
        }
    },

    "Steps": [
        {
        "Name": "GetOrCreateModelFromRegistry",
        "Type": "Lambda",
        "Arguments": {
            "model_package_group_name": "abalone",
            "model_package_version": {
            "Get": "Parameters.ModelPackageVersion"
            }
        },
        "FunctionArn": "arn:aws:lambda:us-east-1:088461143167:function:GetOrCreateModelFromRegistry-abalone",
        "OutputParameters": [
            {
            "OutputName": "model_name",
            "OutputType": "String"
            },
            {
            "OutputName": "model_package_arn",
            "OutputType": "String"
            }
        ]
        },
        {
        "Name": "BaselineChoice",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
            {
                "Type": "Equals",
                "LeftValue": {
                "Get": "Parameters.RegisterNewBaseline"
                },
                "RightValue": true
            }
            ],
            "IfSteps": [
            {
                "Name": "PrepBaselineSetsStep",
                "Type": "Lambda",
                "Arguments": {
                "baseline_file": {
                    "Get": "Parameters.BaselineFile"
                },
                "target_name": "rings",
                "target_type": "float",
                "baseline_X_file_dest_dir": "s3://omm-test-bucket/pipelines/abalone/baseline"
                },
                "FunctionArn": "arn:aws:lambda:us-east-1:088461143167:function:PrepBaselineSetsStep-abalone",
                "OutputParameters": [
                {
                    "OutputName": "baseline_X_dir",
                    "OutputType": "String"
                },
                {
                    "OutputName": "baseline_X_filename",
                    "OutputType": "String"
                }
                ]
            }
            ],
            "ElseSteps": []
        },
        "DependsOn": [
            "GetOrCreateModelFromRegistry"
        ]
        },
        {
        "Name": "ActionTypeChoice",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
            {
                "Type": "Equals",
                "LeftValue": {
                "Get": "Parameters.Action"
                },
                "RightValue": "inference"
            }
            ],
            "IfSteps": [
            {
                "Name": "DeployEndpointStep",
                "Type": "Lambda",
                "Arguments": {
                "model_name": {
                    "Get": "Steps.GetOrCreateModelFromRegistry.OutputParameters['model_name']"
                },
                "model_package_group_name": "abalone",
                "model_package_version_param": {
                    "Get": "Parameters.ModelPackageVersion"
                },
                "data_capture_dir": [
                    "s3://omm-test-bucket/pipelines/abalone/capture"
                ]
                },
                "DependsOn": [
                "ActionTypeChoice"
                ],
                "FunctionArn": "arn:aws:lambda:us-east-1:088461143167:function:DeployEndpointStep-abalone",
                "OutputParameters": [
                {
                    "OutputName": "endpoint_name",
                    "OutputType": "String"
                }
                ]
            },
            {
                "Name": "DataQualityChoice",
                "Type": "Condition",
                "Arguments": {
                "Conditions": [
                    {
                    "Type": "Equals",
                    "LeftValue": {
                        "Get": "Parameters.EnableDataQualityMonitoring"
                    },
                    "RightValue": true
                    }
                ],
                "IfSteps": [
                    {
                    "Name": "ScheduledDataQualityMonitorStep",
                    "Type": "Lambda",
                    "Arguments": {
                        "name": "DataQualityMonitor-abalone",
                        "role_arn": "arn:aws:iam::088461143167:role/SageMakerExecutionRole-1",
                        "deploy_type": "realtime",
                        "monitor_dir": "s3://omm-test-bucket/pipelines/abalone/data-quality",
                        "image_uri": "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
                        "instance_count": 1,
                        "instance_type": {
                        "Get": "Parameters.MonitorInstanceType"
                        },
                        "volume_size_in_gb": 20,
                        "max_runtime_in_seconds": 1800,
                        "dataset_format": {
                        "Csv": {
                            "Header": true
                        }
                        },
                        "schedule_expression": "cron(0 * ? * * *)",
                        "data_analysis_start_time": "-PT2H",
                        "data_analysis_end_time": "-PT1H",
                        "endpoint_name": {
                        "Get": "Steps.DeployEndpointStep.OutputParameters['endpoint_name']"
                        },
                        "data_capture_dir": [
                        "s3://omm-test-bucket/pipelines/abalone/capture"
                        ]
                    },
                    "FunctionArn": "arn:aws:lambda:us-east-1:088461143167:function:ScheduledDataQualityMonitorStep-abalone",
                    "OutputParameters": []
                    }
                ],
                "ElseSteps": []
                },
                "DependsOn": [
                "DeployEndpointStep"
                ]
            },
            {
                "Name": "ModelBiasChoice",
                "Type": "Condition",
                "Arguments": {
                "Conditions": [
                    {
                    "Type": "Equals",
                    "LeftValue": {
                        "Get": "Parameters.EnableModelBiasMonitoring"
                    },
                    "RightValue": true
                    }
                ],
                "IfSteps": [
                    {
                    "Name": "ScheduledModelBiasMonitorStep",
                    "Type": "Lambda",
                    "Arguments": {
                        "name": "ModelBiasMonitor-abalone",
                        "role_arn": "arn:aws:iam::088461143167:role/SageMakerExecutionRole-1",
                        "deploy_type": "realtime",
                        "monitor_dir": "s3://omm-test-bucket/pipelines/abalone/model-bias",
                        "ground_truth_dir": {
                        "Get": "Parameters.GroundTruthDir"
                        },
                        "image_uri": "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
                        "instance_count": 1,
                        "instance_type": {
                        "Get": "Parameters.MonitorInstanceType"
                        },
                        "volume_size_in_gb": 20,
                        "max_runtime_in_seconds": 1800,
                        "dataset_format": {
                        "Csv": {
                            "Header": true
                        }
                        },
                        "schedule_expression": "cron(0 * ? * * *)",
                        "data_analysis_start_time": "-PT2H",
                        "data_analysis_end_time": "-PT1H",
                        "endpoint_name": {
                        "Get": "Steps.DeployEndpointStep.OutputParameters['endpoint_name']"
                        },
                        "data_capture_dir": [
                        "s3://omm-test-bucket/pipelines/abalone/capture"
                        ]
                    },
                    "FunctionArn": "arn:aws:lambda:us-east-1:088461143167:function:ScheduledModelBiasMonitorStep-abalone",
                    "OutputParameters": []
                    }
                ],
                "ElseSteps": []
                },
                "DependsOn": [
                "DeployEndpointStep"
                ]
            },
            {
                "Name": "ModelExplainabilityChoice",
                "Type": "Condition",
                "Arguments": {
                "Conditions": [
                    {
                    "Type": "Equals",
                    "LeftValue": {
                        "Get": "Parameters.EnableModelExplainabilityMonitoring"
                    },
                    "RightValue": true
                    }
                ],
                "IfSteps": [
                    {
                    "Name": "ScheduledModelExplainabilityMonitorStep",
                    "Type": "Lambda",
                    "Arguments": {
                        "name": "ModelExplainabilityMonitor-abalone",
                        "role_arn": "arn:aws:iam::088461143167:role/SageMakerExecutionRole-1",
                        "deploy_type": "realtime",
                        "monitor_dir": "s3://omm-test-bucket/pipelines/abalone/model-explainability",
                        "image_uri": "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
                        "instance_count": 1,
                        "instance_type": {
                        "Get": "Parameters.MonitorInstanceType"
                        },
                        "volume_size_in_gb": 20,
                        "max_runtime_in_seconds": 1800,
                        "dataset_format": {
                        "Csv": {
                            "Header": true
                        }
                        },
                        "schedule_expression": "cron(0 * ? * * *)",
                        "data_analysis_start_time": "-PT2H",
                        "data_analysis_end_time": "-PT1H",
                        "endpoint_name": {
                        "Get": "Steps.DeployEndpointStep.OutputParameters['endpoint_name']"
                        },
                        "data_capture_dir": [
                        "s3://omm-test-bucket/pipelines/abalone/capture"
                        ]
                    },
                    "FunctionArn": "arn:aws:lambda:us-east-1:088461143167:function:ScheduledModelExplainabilityMonitorStep-abalone",
                    "OutputParameters": []
                    }
                ],
                "ElseSteps": []
                },
                "DependsOn": [
                "DeployEndpointStep"
                ]
            },
            {
                "Name": "ModelQualityChoice",
                "Type": "Condition",
                "Arguments": {
                "Conditions": [
                    {
                    "Type": "Equals",
                    "LeftValue": {
                        "Get": "Parameters.EnableModelQualityMonitoring"
                    },
                    "RightValue": true
                    }
                ],
                "IfSteps": [
                    {
                    "Name": "ScheduledModelQualityMonitorStep",
                    "Type": "Lambda",
                    "Arguments": {
                        "name": "ModelQualityMonitor-abalone",
                        "role_arn": "arn:aws:iam::088461143167:role/SageMakerExecutionRole-1",
                        "deploy_type": "realtime",
                        "problem_type": "Regression",
                        "ground_truth_label": "rings",
                        "monitor_dir": "s3://omm-test-bucket/pipelines/abalone/model-quality",
                        "ground_truth_dir": {
                        "Get": "Parameters.GroundTruthDir"
                        },
                        "image_uri": "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer",
                        "instance_count": 1,
                        "instance_type": {
                        "Get": "Parameters.MonitorInstanceType"
                        },
                        "volume_size_in_gb": 20,
                        "max_runtime_in_seconds": 1800,
                        "dataset_format": {
                        "Csv": {
                            "Header": true
                        }
                        },
                        "schedule_expression": "cron(0 * ? * * *)",
                        "data_analysis_start_time": "-PT2H",
                        "data_analysis_end_time": "-PT1H",
                        "endpoint_name": {
                        "Get": "Steps.DeployEndpointStep.OutputParameters['endpoint_name']"
                        },
                        "data_capture_dir": [
                        "s3://omm-test-bucket/pipelines/abalone/capture"
                        ]
                    },
                    "FunctionArn": "arn:aws:lambda:us-east-1:088461143167:function:ScheduledModelQualityMonitorStep-abalone",
                    "OutputParameters": []
                    }
                ],
                "ElseSteps": []
                },
                "DependsOn": [
                "DeployEndpointStep"
                ]
            }
            ],
            "ElseSteps": []
        },
        "DependsOn": [
            "BaselineChoice"
        ]
        }
    ]
}
"""

sm_client=boto3.client('sagemaker')


sm_client.create_pipeline(
    PipelineName='fake-pipe',
    PipelineDefinition=json_definition,
    RoleArn='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1'
)
