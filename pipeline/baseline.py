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
            depends_on=depends_on
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
            depends_on=depends_on
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
            depends_on=depends_on
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
            depends_on=depends_on
        )

        return me_baseline_step