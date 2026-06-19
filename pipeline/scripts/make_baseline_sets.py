import boto3, logging, io
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

def parse_s3_uri(s3_uri):
    # s3://bucket-name/path/to/key
    s3_uri = s3_uri.replace('s3://', '')
    bucket, key = s3_uri.split('/', 1)
    return bucket, key

def df_to_s3(df, s3_uri, index=False, header=True):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=index, header=header)

    bucket, key = parse_s3_uri(s3_uri)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())


def make_baseline_sets(
    baseline_file,
    baseline_pred_file,
    dq_monitor_dir,
    db_monitor_dir,
    mq_monitor_dir,
    mb_monitor_dir,
    me_monitor_dir,
    target_name,
    prediction_name,
    train_file,
    train_X_file,
    target_type=float
):

    baseline=pd.read_csv(baseline_file, header=0)
    baseline_pred=pd.read_csv(baseline_pred_file, header=None)
    baseline_pred.columns=[prediction_name]
    baseline_full = pd.concat([baseline_pred, baseline], axis=1)
    baseline_full[target_name] = baseline_full[target_name].astype(target_type)
    baseline_full[prediction_name] = baseline_full[prediction_name].astype(target_type)

    # Data Quality → input features only
    df_to_s3(
        baseline_full.drop(columns=[target_name, prediction_name]), 
        f'{dq_monitor_dir}/baseline.csv', 
        index=False, 
        header=True
    )
    # baseline_full.drop(columns=[target_name, prediction_name]).to_csv(f'{dq_monitor_dir}/baseline.csv', index=False, header=True)

    # Data Bias → input features + target
    df_to_s3(
        baseline_full.drop(columns=[prediction_name]), 
        f'{db_monitor_dir}/baseline.csv', 
        index=False, 
        header=True
    )
    # baseline_full.drop(columns=[prediction_name]).to_csv(f'{db_monitor_dir}/baseline.csv', index=False, header=True)

    # Model Quality → predictions + ground truth labels
    df_to_s3(
        baseline_full[[target_name, prediction_name]], 
        f'{mq_monitor_dir}/baseline.csv', 
        index=False, 
        header=True
    )
    # baseline_full[[target_name, prediction_name]].to_csv(f'{mq_monitor_dir}/baseline.csv', index=False, header=True)

    # Model Bias → features + predictions + labels
    df_to_s3(
        baseline_full, 
        f'{mb_monitor_dir}/baseline.csv', 
        index=False, 
        header=True
    )
    # baseline_full.to_csv(f'{mb_monitor_dir}/baseline.csv', index=False, header=True)

    # Model Explainability → input features + predictions (uses SHAP values)
    df_to_s3(
        baseline_full.drop(columns=[target_name]), 
        f'{me_monitor_dir}/baseline.csv', 
        index=False, 
        header=True
    )
    # baseline_full.drop(columns=[target_name]).to_csv(f'{me_monitor_dir}/baseline.csv', index=False, header=True)

    #
    train=pd.read_csv(train_file, header=None)
    train_X = train.iloc[:, 1:]

    df_to_s3(
        train_X, 
        train_X_file, 
        index=False, 
        header=False
    )
    # train_X.to_csv(train_X_file, index=False, header=False)

    return None


def handler(event, context):
    baseline_file = event['baseline_file']
    baseline_pred_file = event['baseline_pred_file']
    dq_monitor_dir = event['dq_monitor_dir']
    db_monitor_dir = event['db_monitor_dir']
    mq_monitor_dir = event['mq_monitor_dir']
    mb_monitor_dir = event['mb_monitor_dir']
    me_monitor_dir = event['me_monitor_dir']
    target_name = event['target_name']
    prediction_name = event['prediction_name']
    train_file = event['train_file']
    train_X_file = event['train_X_file']
    target_type = event['target_type'] if 'target_type' in event else float

    result = make_baseline_sets(
        baseline_file,
        baseline_pred_file,
        dq_monitor_dir,
        db_monitor_dir,
        mq_monitor_dir,
        mb_monitor_dir,
        me_monitor_dir,
        target_name,
        prediction_name,
        train_file,
        train_X_file,
        target_type=target_type
    )
    
    return {'result': result}