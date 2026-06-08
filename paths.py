data_bucket='omm-test-bucket'
project_path = 'models/abalone'

train_dir=     f's3://{data_bucket}/{project_path}/data/input/train'
validation_dir=f's3://{data_bucket}/{project_path}/data/input/validation'
test_dir=      f's3://{data_bucket}/{project_path}/data/input/test'
baseline_dir=  f's3://{data_bucket}/{project_path}/data/baseline'

# Training
train_file=     f'{train_dir}/train.csv'
validation_file=f'{validation_dir}/validation.csv'
test_file=      f'{test_dir}/test.csv'
test_X_file=    f'{test_dir}/test_X.csv'
test_y_file=    f'{test_dir}/test_y.csv'
baseline_file=  f'{baseline_dir}/baseline.csv'
baseline_X_file=f'{baseline_dir}/baseline_X.csv'
batch_in_dir=   f's3://{data_bucket}/{project_path}/data/batch-input'
batch_out_dir=  f's3://{data_bucket}/{project_path}/data/batch-output'
baseline_model_out_file= f'{batch_out_dir}/baseline_X.csv.out'
baseline_pred_file=f'{baseline_dir}/baseline_pred.csv'
model_dir=      f's3://{data_bucket}/{project_path}/model'

# Deploy
data_capture_dir=  f's3://{data_bucket}/{project_path}/data/capture'
ground_truth_dir=  f's3://{data_bucket}/{project_path}/data/ground-truth'

# Monitors
monitors_dir=  f's3://{data_bucket}/{project_path}/data/monitors'
dq_monitor_dir=f'{monitors_dir}/data-quality'
mq_monitor_dir=f'{monitors_dir}/model-quality'
mb_monitor_dir=f'{monitors_dir}/model-bias'
me_monitor_dir=f'{monitors_dir}/model-explainability'