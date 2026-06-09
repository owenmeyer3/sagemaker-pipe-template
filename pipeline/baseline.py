import pandas as pd
import sagemaker

class Baseliner():

    def __init__(self, model_name, data_dir_uri, baseline_file, train_file, monitor_instance_type):
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
        
        self.monitor_instance_type=monitor_instance_type

    def make_baseline_sets(self, target_name, prediction_name, target_type=float):

        baseline=pd.read_csv(self.baseline_file, header=0)
        baseline_pred=pd.read_csv(self.baseline_pred_file, header=None)
        baseline_pred.columns=[self.prediction_name]
        baseline_full = pd.concat([self.baseline_pred, baseline], axis=1)
        baseline_full[target_name] = baseline_full[target_name].astype(target_type)
        baseline_full[prediction_name] = baseline_full[prediction_name].astype(target_type)

        # Data Quality    → input features only
        baseline_full.drop(columns=[target_name, prediction_name]).to_csv(f'{self.dq_monitor_dir}/baseline.csv', index=False, header=True)

        # Model Quality   → predictions + ground truth labels
        baseline_full[[target_name, prediction_name]].to_csv(f'{self.mq_monitor_dir}/baseline.csv', index=False, header=True)

        # Model Bias      → features + predictions + labels
        baseline_full.to_csv(f'{self.mb_monitor_dir}/baseline.csv', index=False, header=True)

        # Model Explainability → input features + predictions (uses SHAP values)
        baseline_full.drop(columns=[target_name]).to_csv(f'{self.me_monitor_dir}/baseline.csv', index=False, header=True)

        train=pd.read_csv(self.train_file, header=None)
        train_X = train.iloc[:, 1:]
        train_X.to_csv(self.train_file, index=False, header=False)


    def get_data_quality_step(self, sagemaker_session, role, depends_on=[]):

        data_quality_monitor = sagemaker.model_monitor.DefaultModelMonitor(
            role=role,
            instance_count=1,
            instance_type=self.monitor_instance_type,
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
            sagemaker_session=sagemaker_session
        )

        dq_baseline_step = sagemaker.workflow.steps.ProcessingStep(
            name='DataQualityBaselineStep',
            step_args=data_quality_monitor.suggest_baseline(
                baseline_dataset=f'{self.dq_monitor_dir}/baseline.csv',
                dataset_format=sagemaker.model_monitor.dataset_format.DatasetFormat.csv(header=True),
                output_s3_uri=f"{self.dq_monitor_dir}/info",
            ),
            depends_on=depends_on,
            sagemaker_session=sagemaker_session
        )

        return dq_baseline_step


    def get_model_quality_step(self, sagemaker_session, role, target_name, prediction_name, depends_on=[]):

        model_quality_monitor = sagemaker.model_monitor.ModelQualityMonitor(
            role=role,
            instance_count=1,
            instance_type=self.monitor_instance_type,
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
            sagemaker_session=sagemaker_session
        )

        mq_baseline_step = sagemaker.workflow.steps.ProcessingStep(
            name='ModelQualityBaselineStep',
            step_args=model_quality_monitor.suggest_baseline(
                baseline_dataset=f'{self.mq_monitor_dir}/baseline.csv',
                dataset_format=sagemaker.model_monitor.dataset_format.DatasetFormat.csv(header=True),
                output_s3_uri=f'{self.mq_monitor_dir}/info',
                problem_type='Regression',
                inference_attribute=prediction_name,   # target column header (named by PySpark)
                ground_truth_attribute=target_name, # output column header (only 1 output)
                wait=False,
                logs=False
            ),
            depends_on=depends_on,
            sagemaker_session=sagemaker_session
        )

        return mq_baseline_step


    def get_model_bias_step(self, sagemaker_session, role, target_name, prediction_name, depends_on=[]):

        model_bias_monitor = sagemaker.model_monitor.ModelBiasMonitor(
            role=role,
            instance_count=1,
            instance_type=self.monitor_instance_type,
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
            sagemaker_session=sagemaker_session
        )

        mb_baseline_step = sagemaker.workflow.steps.ProcessingStep(
            name='ModelBiasBaselineStep',
            step_args=model_bias_monitor.suggest_baseline(
                data_config=sagemaker.clarify.DataConfig(
                    s3_data_input_path=f'{self.mb_monitor_dir}/baseline.csv',
                    s3_output_path=f'{self.mb_monitor_dir}/info',
                    dataset_type = 'text/csv',
                    label=target_name,
                    predicted_label=prediction_name, 
                ),
                bias_config=sagemaker.clarify.BiasConfig(facet_name='sex_F', label_values_or_threshold=[7], facet_values_or_threshold=[0.5]),
                model_config=sagemaker.clarify.ModelConfig(
                    model_name=self.model_name,
                    instance_type=self.monitor_instance_type,
                    instance_count=1,
                    accept_type='text/csv',
                    content_type='text/csv'
                ),
                # model_predicted_label_config=sagemaker.clarify.ModelPredictedLabelConfig(
                #     probability_threshold=0.5  # threshold to convert float prediction to binary label
                # ), 
                wait=False,
                logs=False
            ),
            depends_on=depends_on,
            sagemaker_session=sagemaker_session
        )

        return mb_baseline_step

    def get_model_explainability_step(self, sagemaker_session, role, depends_on=[]):
        model_explainability_monitor = sagemaker.model_monitor.ModelExplainabilityMonitor(
            role=role,
            instance_count=1,
            instance_type=self.monitor_instance_type,
            sagemaker_session=sagemaker_session
        )

        train_X=pd.read_csv(self.train_X_file, header=None)

        me_baseline_step = sagemaker.workflow.steps.ProcessingStep(
            name='ExplainabilityBaselineStep',
            step_args=model_explainability_monitor.suggest_baseline(
                data_config=sagemaker.clarify.DataConfig(
                    s3_data_input_path=f'{self.me_monitor_dir}/baseline.csv',
                    s3_output_path=f'{self.me_monitor_dir}/info',
                    dataset_type='text/csv'
                ),
                model_config=sagemaker.clarify.ModelConfig(
                    model_name=self.model_name,
                    instance_type=self.monitor_instance_type,
                    instance_count=1,
                    accept_type='text/csv',
                    content_type='text/csv'
                ),
                explainability_config=sagemaker.clarify.SHAPConfig(
                    baseline=[train_X.mean().tolist()],  # mean of training features as baseline
                    num_samples=100,
                    agg_method='mean_abs'
                )
            ),
            depends_on=depends_on,
            sagemaker_session=sagemaker_session
        )

        return me_baseline_step
    
    def get_monitor_batch_transform_step(self, sagemaker_session, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None, depends_on=[]):
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

        batch_transform_input = sagemaker.core.shapes.shapes.BatchTransformInput(
            data_captured_destination_s3_uri=f'{self.data_capture_dir}',
            dataset_format=sagemaker.core.shapes.shapes.MonitoringDatasetFormat(csv=sagemaker.core.shapes.shapes.MonitoringCsvDatasetFormat(header=True)),
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
            sagemaker_session=sagemaker_session
        )

        monitoring_resources = sagemaker.core.shapes.shapes.MonitoringResources(
            cluster_config=sagemaker.core.shapes.shapes.MonitoringClusterConfig(
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=5,
                volume_kms_key_id=None
            )
        )
        monitoring_app_specification=sagemaker.core.shapes.shapes.MonitoringAppSpecification(
            image_uri=sagemaker.core.image_uris.retrieve(framework='model-monitor', region='us-east-1'), # required - the container to run
            # container_entrypoint=['...'],       # optional - override entrypoint
            # container_arguments=['...'],        # optional - override arguments
            # record_preprocessor_source_uri='s3://...', # optional - preprocessing script
            # post_analytics_processor_source_uri='s3://...' # optional - postprocessing script
        )

        monitoring_job_definition = sagemaker.core.shapes.shapes.MonitoringJobDefinition(
            monitoring_inputs= [sagemaker.core.shapes.shapes.MonitoringInput(batch_transform_input=batch_transform_input)], 
            monitoring_output_config=sagemaker.core.shapes.shapes.MonitoringOutputConfig(
                monitoring_outputs=[
                    sagemaker.core.shapes.shapes.MonitoringOutput(
                        s3_output=sagemaker.core.shapes.shapes.MonitoringS3Output(
                            local_path='/opt/ml/processing/input', 
                            s3_uri='s3://omm-test-bucket/models/test/batch-output/'
                        )
                    )]
            ),
            monitoring_resources=monitoring_resources, 
            monitoring_app_specification=monitoring_app_specification, 
            role_arn=sagemaker.core.helper.session_helper.get_execution_role(), 
            stopping_condition=sagemaker.core.shapes.shapes.MonitoringStoppingCondition(
                max_runtime_in_seconds=400
            ), 
            environment={}, 
            # network_config: NetworkConfig | None = Unassigned()
        )

        schedule_config = sagemaker.core.shapes.shapes.ScheduleConfig(
            schedule_expression='cron(0 * ? * * *)', 
            data_analysis_start_time="-PT1H", 
            data_analysis_end_time="-PT2H"
            )

#################### MODEL QUALITY

    def get_job_definition(self, sagemaker_session, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None):
        batch_transform_input = sagemaker.core.shapes.shapes.BatchTransformInput(
            data_captured_destination_s3_uri=f'{self.data_capture_dir}',
            dataset_format=sagemaker.core.shapes.shapes.MonitoringDatasetFormat(csv=sagemaker.core.shapes.shapes.MonitoringCsvDatasetFormat(header=True)),
            local_path='/opt/ml/processing/input',
            s3_input_mode='File',
            s3_data_distribution_type='FullyReplicated', 
            features_attribute=','.join(self.features),
            inference_attribute=self.target,
            probability_attribute=probability_attribute, 
            probability_threshold_attribute=probability_threshold_attribute, 
            exclude_features_attribute=exclude_features_attribute,
            start_time_offset="-PT2H",
            end_time_offset="-PT1H",
            sagemaker_session=sagemaker_session
        )

        monitoring_resources = sagemaker.core.shapes.shapes.MonitoringResources(
            cluster_config=sagemaker.core.shapes.shapes.MonitoringClusterConfig(
                instance_count=1,
                instance_type=self.monitor_instance_type,
                volume_size_in_gb=5,
                volume_kms_key_id=None
            )
        )
        monitoring_app_specification=sagemaker.core.shapes.shapes.MonitoringAppSpecification(
            image_uri=sagemaker.core.image_uris.retrieve(framework='model-monitor', region='us-east-1'), # required - the container to run
            # container_entrypoint=['...'],       # optional - override entrypoint
            # container_arguments=['...'],        # optional - override arguments
            # record_preprocessor_source_uri='s3://...', # optional - preprocessing script
            # post_analytics_processor_source_uri='s3://...' # optional - postprocessing script
        )

        monitoring_job_definition = sagemaker.core.shapes.shapes.MonitoringJobDefinition(
            monitoring_inputs= [sagemaker.core.shapes.shapes.MonitoringInput(batch_transform_input=batch_transform_input)], 
            monitoring_output_config=sagemaker.core.shapes.shapes.MonitoringOutputConfig(
                monitoring_outputs=[
                    sagemaker.core.shapes.shapes.MonitoringOutput(
                        s3_output=sagemaker.core.shapes.shapes.MonitoringS3Output(
                            local_path='/opt/ml/processing/input', 
                            s3_uri='s3://omm-test-bucket/models/test/batch-output/'
                        )
                    )]
            ),
            monitoring_resources=monitoring_resources, 
            monitoring_app_specification=monitoring_app_specification, 
            role_arn=sagemaker.core.helper.session_helper.get_execution_role(), 
            stopping_condition=sagemaker.core.shapes.shapes.MonitoringStoppingCondition(
                max_runtime_in_seconds=400
            ), 
            environment={}, 
            # network_config: NetworkConfig | None = Unassigned()
        )
    
        return monitoring_job_definition

        schedule_config = sagemaker.core.shapes.shapes.ScheduleConfig(
            schedule_expression='cron(0 * ? * * *)', 
            data_analysis_start_time="-PT1H", 
            data_analysis_end_time="-PT2H"
            )

    def get_batch_model_quality_step(self, sagemaker_session, transform_step, schedule_config, monitoring_job_definition, depends_on=[]):

        model_dashboard_monitoring_schedule=sagemaker.core.shapes.shapes.ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=sagemaker.core.shapes.shapes.MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="'ModelQualityJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="ModelQuality" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )

        model_quality_check_config=sagemaker.mlops.workflow.quality_check_step.ModelQualityCheckConfig(
            baseline_dataset=f'{self.mq_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            problem_type='Regression',
            output_s3_uri=f'{self.mq_monitor_dir}/info'
        )
        
        mq_monitor_step = sagemaker.workflow.monitor_batch_transform_step.MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=model_quality_check_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.mq_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.mq_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.mq_monitor_dir}/transforms',
            ground_truth_input='s3://omm-test-bucket/models/abalone/data/ground-truth/',  # ground truth labels
            fail_on_violation=False,
            depends_on=depends_on,
            sagemaker_session=sagemaker_session
        )

        return mq_monitor_step

#################### MODEL BIAS
    def get_batch_model_bias_step(self, sagemaker_session, transform_step, role, schedule_config, monitoring_job_definition, depends_on=[]):

        model_dashboard_monitoring_schedule=sagemaker.core.shapes.shapes.ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=sagemaker.core.shapes.shapes.MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="ModelBiasJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="ModelBias" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )
                
        model_bias_check_config=sagemaker.mlops.workflow.quality_check_step.ModelQualityCheckConfig(
            baseline_dataset=f'{self.mb_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            problem_type='Regression',
            output_s3_uri=f'{self.mb_monitor_dir}/predictions'
        )
        
        mb_monitor_step = sagemaker.workflow.monitor_batch_transform_step.MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=model_bias_check_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.mb_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.mb_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.mb_monitor_dir}/transforms',
            ground_truth_input='s3://omm-test-bucket/models/abalone/data/ground-truth/',  # ground truth labels
            fail_on_violation=False,
            sagemaker_session=sagemaker_session
        )
        return mb_monitor_step

#################### DATA QUALITY
    def get_batch_data_quality_step(self, sagemaker_session, transform_step, role, schedule_config, monitoring_job_definition, depends_on=[]):

        model_dashboard_monitoring_schedule=sagemaker.core.shapes.shapes.ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=sagemaker.core.shapes.shapes.MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="DataQualityJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="DataQuality" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )
              
        data_quality_config=sagemaker.mlops.workflow.quality_check_step.DataQualityCheckConfig(
                baseline_dataset=f'{self.dq_monitor_dir}/baseline.csv', 
                dataset_format={}, 
                output_s3_uri=f'{self.dq_monitor_dir}/predictions'
        )
        
        dq_monitor_step = sagemaker.workflow.monitor_batch_transform_step.MonitorBatchTransformStep(
            name='DataQualityMonitorStep',
            monitor_configuration=data_quality_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.dq_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.dq_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.dq_monitor_dir}/transforms',
            ground_truth_input='s3://omm-test-bucket/models/abalone/data/ground-truth/',  # ground truth labels
            fail_on_violation=False,
            sagemaker_session=sagemaker_session
        )

        return dq_monitor_step

#################### DATA BIAS
    def get_batch_data_bias_step(self, sagemaker_session, transform_step, role, schedule_config, monitoring_job_definition, depends_on=[]):

        model_dashboard_monitoring_schedule=sagemaker.core.shapes.shapes.ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=sagemaker.core.shapes.shapes.MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="DataBiasJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="DataBias" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )
                
        data_bias_check_config=sagemaker.mlops.workflow.quality_check_step.ModelQualityCheckConfig(
            baseline_dataset=f'{self.db_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            problem_type='Regression',
            output_s3_uri=f'{self.db_monitor_dir}/predictions'
        )
        
        mq_monitor_step = sagemaker.workflow.monitor_batch_transform_step.MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=data_bias_check_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.db_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.db_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.db_monitor_dir}/transforms',
            ground_truth_input='s3://omm-test-bucket/models/abalone/data/ground-truth/',  # ground truth labels
            fail_on_violation=False,
            sagemaker_session=sagemaker_session
        )

#################### EXPLAINABILITY
    def get_batch_model_explainabilty_step(self, sagemaker_session, transform_step, role, schedule_config, monitoring_job_definition, depends_on=[]):

        model_dashboard_monitoring_schedule=sagemaker.core.shapes.shapes.ModelDashboardMonitoringSchedule(
            batch_transform_input=transform_step.arguments['TransformInput'],
            monitoring_schedule_config=sagemaker.core.shapes.shapes.MonitoringScheduleConfig(
                schedule_config=schedule_config,
                monitoring_job_definition_name="ModelExplainabilityJobDefinition",
                monitoring_job_definition=monitoring_job_definition,
                monitoring_type="ModelExplainability" #DataQuality | ModelQuality | ModelBias | ModelExplainability
            )
        )
                
        model_explainabilty_check_config=sagemaker.mlops.workflow.quality_check_step.ModelQualityCheckConfig(
            baseline_dataset=f'{self.me_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            problem_type='Regression',
            output_s3_uri=f'{self.me_monitor_dir}/predictions'
        )
        
        me_monitor_step = sagemaker.workflow.monitor_batch_transform_step.MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=model_explainabilty_check_config,
            transform_step_args=transform_step.arguments,
            baseline_statistics=f'{self.me_monitor_dir}/info/statistics.json',
            baseline_constraints=f'{self.me_monitor_dir}/info/constraints.json',
            output_s3_uri=f'{self.me_monitor_dir}/transforms',
            ground_truth_input='s3://omm-test-bucket/models/abalone/data/ground-truth/',  # ground truth labels
            fail_on_violation=False,
            sagemaker_session=sagemaker_session
        )

        return me_monitor_step

#################### 

        # model_quality_monitor = sagemaker.model_monitor.ModelQualityMonitor(
        #     role=role,
        #     instance_count=1,
        #     instance_type=self.monitor_instance_type,
        #     volume_size_in_gb=20,
        #     max_runtime_in_seconds=1800,
        #     sagemaker_session=sagemaker_session
        # )

        data_quality_config=sagemaker.mlops.workflow.quality_check_step.DataQualityCheckConfig(
                baseline_dataset=f'{self.dq_monitor_dir}/baseline.csv', 
                dataset_format={}, 
                output_s3_uri=f'{self.dq_monitor_dir}/predictions'
        )

        model_quality_check_config=sagemaker.mlops.workflow.quality_check_step.ModelQualityCheckConfig(
                baseline_dataset=f'{self.mq_monitor_dir}/baseline.csv', 
                dataset_format={}, 
                problem_type='Regression'
                output_s3_uri=f'{self.mq_monitor_dir}/predictions'
            )
        
        data_bias_check_config=sagemaker.mlops.workflow.clarify_check_step.DataBiasCheckConfig(
            baseline_dataset=f'{self.mq_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            output_s3_uri=f'{self.mq_monitor_dir}/predictions'
        )
    
        model_bias_check_config=sagemaker.mlops.workflow.clarify_check_step.ModelBiasCheckConfig(
            baseline_dataset=f'{self.mb_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            output_s3_uri=f'{self.mb_monitor_dir}/predictions'
        )
    
        model_explainability_check_config=sagemaker.mlops.workflow.clarify_check_step.ModelExplainabilityCheckConfig(
            baseline_dataset=f'{self.me_monitor_dir}/baseline.csv', 
            dataset_format={}, 
            output_s3_uri=f'{self.me_monitor_dir}/predictions'
        )
    

        clarify_check_config=sagemaker.mlops.workflow.clarify_check_step.ClarifyCheckConfig(
            data_config=sagemaker.clarify.DataConfig(
                s3_data_input_path=f'{self.mq_monitor_dir}/baseline.csv',
                s3_output_path=f'{self.mq_monitor_dir}/info',
                dataset_type='text/csv'
                ), 
                monitoring_analysis_config_uri=None
            )
        
        check_job_config=sagemaker.mlops.workflow.check_job_config.CheckJobConfig(
            role, 
            instance_count=1, 
            instance_type=self.monitor_instance_type, 
            volume_size_in_gb=30, 
            volume_kms_key=None, 
            output_kms_key=None, 
            max_runtime_in_seconds=None, 
            base_job_name=None, 
            sagemaker_session=sagemaker_session, 
            env=None, 
            tags=None, 
            network_config=None
        )

        transformer = sagemaker.transformer.Transformer(
            model_name=create_model_step.properties.Outputs['model_name'],
            instance_count=1,
            instance_type=self.monitor_instance_type,
            output_path=f'{self.mbt_monitor_dir}/transformations',
            accept='text/csv',
            assemble_with='Line',
            sagemaker_session=sagemaker_session
        )

        transform_step = sagemaker.mlops.workflow.steps.TransformStep(
            name="TransformStep",
            display_name="TransformStep",
            description="",
            step_args=transformer.transform(
                data=input_data.default_value,
                content_type='text/csv',
                split_type='Line'
            ),
        )

        quality_check_step = sagemaker.mlops.workflow.quality_check_step.QualityCheckStep(
            name="QualityCheckStep",
            description="",
            quality_check_config=quality_check_config,
            check_job_config=check_job_config,
            skip_check=False,
            supplied_baseline_statistics=f'{self.mq_monitor_dir}/info/supplied_baseline_statistics.json', 
            supplied_baseline_constraints=f'{self.mq_monitor_dir}/info/constraints.json', 
            fail_on_violation=True
        )

        clarify_check_step = sagemaker.mlops.workflow.clarify_check_step.ClarifyCheckStep(
            name="ClarifyCheckStep",
            description="",
            quality_check_config=clarify_check_config,
            check_job_config=check_job_config,
            skip_check=False,
            supplied_baseline_statistics=f'{self.mq_monitor_dir}/info/supplied_baseline_statistics.json', 
            supplied_baseline_constraints=f'{self.mq_monitor_dir}/info/constraints.json', 
            fail_on_violation=True
        )
        
        # Quality Check
        #   monitor_before_transform (bool): If to run data quality or model explainability
        #    monitoring type, a true value of this flag indicates
        #    running the check step before the transform job.
        sagemaker.mlops.workflow.monitor_batch_transform_step.MonitorBatchTransformStep(
            'MonitorBatchTransformStep', 
            transform_step_args=transform_step_args,
            monitor_configuration=quality_check_config, 
            check_job_configuration=check_job_config, 
            monitor_before_transform=False, 
            fail_on_violation=True, 
            supplied_baseline_statistics=f'{self.mq_monitor_dir}/info/supplied_baseline_statistics.json', 
            supplied_baseline_constraints=f'{self.mq_monitor_dir}/info/constraints.json', 

            )
        
        # Clarify Check
        



            
        transformer = sagemaker.transformer.Transformer(
            model_name=create_model_step.properties.Outputs['model_name'],
            instance_count=1,
            instance_type=self.monitor_instance_type,
            output_path=f'{self.mbt_monitor_dir}/info',
            accept='text/csv',
            assemble_with='Line',
            sagemaker_session=sagemaker_session
        )
        mq_monitor_step = sagemaker.workflow.monitor_batch_transform_step.MonitorBatchTransformStep(
            name='ModelQualityMonitorStep',
            monitor_configuration=model_quality_monitor,
            transform_step_args=transform_step_args,
            
            baseline_statistics=read_ssm_step.properties.Outputs['mq_statistics_path'],
            baseline_constraints=read_ssm_step.properties.Outputs['mq_constraints_path'],
            output_s3_uri=f'{output_path.default_value}/mq-reports/',
            ground_truth_input=ground_truth_s3_path,  # ground truth labels
            fail_on_violation=False
        )

        mbt_baseline_step = sagemaker.workflow.steps.ProcessingStep(
            name='ExplainabilityBaselineStep',
            step_args=model_explainability_monitor.suggest_baseline(
                data_config=sagemaker.clarify.DataConfig(
                    s3_data_input_path=f'{self.mbt_monitor_dir}/baseline.csv',
                    s3_output_path=f'{self.mbt_monitor_dir}/info',
                    dataset_type='text/csv'
                ),
                model_config=sagemaker.clarify.ModelConfig(
                    model_name=self.model_name,
                    instance_type=self.monitor_instance_type,
                    instance_count=1,
                    accept_type='text/csv',
                    content_type='text/csv'
                ),
                explainability_config=sagemaker.clarify.SHAPConfig(
                    baseline=[train_X.mean().tolist()],  # mean of training features as baseline
                    num_samples=100,
                    agg_method='mean_abs'
                )
            ),
            depends_on=depends_on
        )
    
        return mbt_baseline_step