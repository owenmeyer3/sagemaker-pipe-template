import utils, datetime, sagemaker, boto3, importlib
import pandas as pd
import numpy as np
import paths as p
from sagemaker.model_monitor.dataset_format import DatasetFormat

data_bucket='omm-test-bucket'
project_path = 'models/abalone'


boto_session=boto3.Session(region_name='us-east-1')
model_package_group_name='abalone'
model_version=1
endpoint_name='abalone-endpoint'
target_name='rings'
prediction_name=target_name+'_prediction'
baseline_file=p.baseline_file
baseline_pred_file=p.baseline_pred_file
data_dir_uri=  f's3://{data_bucket}/{project_path}/data'
# - capture/
# - monitors/
#   - data-quality/
#     - baseline.csv
#   - model-quality/
#     - baseline.csv
#   - model-bias/
#     - baseline.csv
#   - model-explainability/
#     - baseline.csv
# - ground-truth/
# - baseline/
#   - baseline.csv (header / features + target)
#   - baseline_pred.csv (headless / target_prediction)

def pre_deploy(boto_session, data_dir_uri, model_package_group_name, model_version, target_name, prediction_name, target_type=float):
    monitors_dir=      f'{data_dir_uri}/monitors'
    baseline_file=     f'{data_dir_uri}/baseline/baseline.csv'
    baseline_pred_file=f'{data_dir_uri}/baseline/baseline_pred.csv'

    baseline=pd.read_csv(baseline_file, header=0)
    baseline_pred=pd.read_csv(baseline_pred_file, header=None)
    baseline_pred.columns=[prediction_name]
    baseline_full = pd.concat([baseline_pred, baseline], axis=1)
    baseline_full[target_name] = baseline_full[target_name].astype(target_type)
    baseline_full[prediction_name] = baseline_full[prediction_name].astype(target_type)

    # Make monitor baseline datasets
    dq_monitor_dir=f'{monitors_dir}/data-quality'
    mq_monitor_dir=f'{monitors_dir}/model-quality'
    mb_monitor_dir=f'{monitors_dir}/model-bias'
    me_monitor_dir=f'{monitors_dir}/model-explainability'

    # Data Quality    → input features only
    baseline_full.drop(columns=[target_name, prediction_name]).to_csv(f'{dq_monitor_dir}/baseline.csv', index=False, header=True)

    # Model Quality   → predictions + ground truth labels
    baseline_full[[target_name, prediction_name]].to_csv(f'{mq_monitor_dir}/baseline.csv', index=False, header=True)

    # Model Bias      → features + predictions + labels
    baseline_full.to_csv(f'{mb_monitor_dir}/baseline.csv', index=False, header=True)

    # Model Explainability → input features + predictions (uses SHAP values)
    baseline_full.drop(columns=[target_name]).to_csv(f'{me_monitor_dir}/baseline.csv', index=False, header=True)

def deploy_realtime(
        boto_session, 
        model_package_group_name, 
        model_version, 
        endpoint_name, 
        data_dir_uri,
        endpoint_instance_type='ml.m5.large', 
    ):

    data_capture_dir=  f'{data_dir_uri}/capture'
    

    # create or get model
    model_name, unused_model_arn = utils.create_model_object_from_registry(boto_session, model_package_group_name, model_version)

    # deploy endpoint
    utils.deploy_model_endpoint(boto_session, model_name, endpoint_name, data_capture_dir, instance_type=endpoint_instance_type)

def deploy_batch(
        boto_session, 
        model_package_group_name, 
        model_version, 
        endpoint_name, 
        data_dir_uri,
        endpoint_instance_type='ml.m5.large', 
    ):
    pass


def monitor_realtime(
        boto_session, 
        model_name,
        endpoint_name, 
        target_name, 
        prediction_name, 
        data_dir_uri,
        monitor_instance_type='ml.m5.large'
    ):

    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    role = sagemaker.get_execution_role(sagemaker_session)

    # Make monitor baseline datasets
    ground_truth_dir=  f'{data_dir_uri}/ground-truth'
    monitors_dir=  f'{data_dir_uri}/monitors'
    dq_monitor_dir=f'{monitors_dir}/data-quality'
    mq_monitor_dir=f'{monitors_dir}/model-quality'
    mb_monitor_dir=f'{monitors_dir}/model-bias'
    me_monitor_dir=f'{monitors_dir}/model-explainability'

    # Schedule data quality monitor
    data_quality_monitor = sagemaker.model_monitor.DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type=monitor_instance_type,
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=sagemaker_session
    )
    data_quality_monitor.suggest_baseline(
        baseline_dataset=f'{dq_monitor_dir}/baseline.csv',
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=f"{dq_monitor_dir}/info",
        wait=True,
        logs=False
    )
    data_quality_monitor.create_monitoring_schedule(
        monitor_schedule_name='abalone-data-quality-monitor',
        endpoint_input=endpoint_name,
        output_s3_uri=f'{dq_monitor_dir}/reports',
        statistics=data_quality_monitor.baseline_statistics(),
        constraints=data_quality_monitor.suggested_constraints(),
        schedule_cron_expression=sagemaker.model_monitor.CronExpressionGenerator.hourly()
    )

    # Schedule model quality monitor
    model_quality_monitor = sagemaker.model_monitor.ModelQualityMonitor(
        role=role,
        instance_count=1,
        instance_type=monitor_instance_type,
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=sagemaker_session
    )
    model_quality_monitor.suggest_baseline(
        baseline_dataset=f'{mq_monitor_dir}/baseline.csv',
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=f'{mq_monitor_dir}/info',
        problem_type='Regression',
        inference_attribute=prediction_name,   # target column header (named by PySpark)
        ground_truth_attribute=target_name, # output column header (only 1 output)
        wait=True,
        logs=False
    )
    model_quality_monitor.create_monitoring_schedule(
        monitor_schedule_name='abalone-model-quality-monitor',
        endpoint_input=sagemaker.model_monitor.EndpointInput(
            endpoint_name=endpoint_name,
            destination='/opt/ml/processing/input/endpoint',
            inference_attribute=prediction_name  # column index of prediction in output
        ),
        ground_truth_input=ground_truth_dir,
        problem_type='Regression',
        output_s3_uri=f'{mq_monitor_dir}/reports',
        constraints=model_quality_monitor.suggested_constraints(),
        schedule_cron_expression=sagemaker.model_monitor.CronExpressionGenerator.hourly()
    )

    # Schedule model bias monitor
    model_bias_monitor = sagemaker.model_monitor.ModelBiasMonitor(
        role=role,
        instance_count=1,
        instance_type=monitor_instance_type,
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=sagemaker_session
    )
    model_bias_monitor.suggest_baseline(
        data_config=sagemaker.clarify.DataConfig(
            s3_data_input_path=f'{mb_monitor_dir}/baseline.csv',
            s3_output_path=f'{mb_monitor_dir}/info',
            dataset_type = 'text/csv',
            label=target_name,
            predicted_label=prediction_name, 
        ),
        bias_config=sagemaker.clarify.BiasConfig(facet_name='sex_F', label_values_or_threshold=[7], facet_values_or_threshold=[0.5]),
        model_config=sagemaker.clarify.ModelConfig(
            model_name=model_name,
            instance_type=monitor_instance_type,
            instance_count=1,
            accept_type='text/csv',
            content_type='text/csv'
        ),
        # model_predicted_label_config=sagemaker.clarify.ModelPredictedLabelConfig(
        #     probability_threshold=0.5  # threshold to convert float prediction to binary label
        # ), 
        wait=True,
        logs=False
    )
    model_bias_monitor.create_monitoring_schedule(
        monitor_schedule_name='abalone-model-bias-monitor',
        endpoint_input=sagemaker.model_monitor.EndpointInput(
            endpoint_name=endpoint_name,
            destination='/opt/ml/processing/input/endpoint',
            inference_attribute='0'
        ),
        ground_truth_input=ground_truth_dir,
        output_s3_uri=f'{mb_monitor_dir}/reports',
        schedule_cron_expression=sagemaker.model_monitor.CronExpressionGenerator.hourly()
    )