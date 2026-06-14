# sagemaker-pipe-template

### Pre-Deploy-Structure
```
bucket/models/project-name/
    - model
        - model-name
            - XXX
    - data/
        - baseline/
            - baseline_pred.csv (1 col, headless)
            - baseline_X.csv (X, headless)
            - baseline.csv (target + features w/ head)
        - input/
            - test/
                - test_X.csv (id + features, headless)
                - test_y.csv (id + target, headless)
            - train/
                - train.csv
            - validation/
                - validation.csv
```

### Post-Deploy-Structure
```
bucket/models/project-name/
    - model
        - model-name
            - XXX
    - data/
        - baseline/
            - baseline_pred.csv (1 col, headless)
            - baseline_X.csv (X, headless)
            - baseline.csv (target + features w/ head)
        - batch-input/
            - XXX
        - batch-output/
            - XXX
        -capture/
            - project-name-endpoint/
                - AllTraffic/YYYY/MM/DD/H24/
                    - id.jsonl
        - ground-truth/
            - YYYY/MM/DD/H24/
                - mm-ss-ffffff.jsonl
        - input/
            - test/
                - test_X.csv (id + features, headless)
                - test_y.csv (id + target, headless)
            - train/
                - train.csv
            - validation/
                - validation.csv
        - monitors/
            - data-bias/
                - info/
                    - analysis_config.json
            - data-quality/
                - baseline.csv
                - info/
                    - constraints.json
                    - statistics.json
            - model-bias/
                - baseline.csv
                - info/
                    - analysis_config.json
                    - analysis.json
                    - report.html
                    - report.ipynb
                    - report.pdf
                - reports/
                    - merge/
                        - XXX
                    - job-def-id/
                        - id/
                            - analysis_config.json
            - model-explainability/
                - baseline.csv
                - test_X.csv (id + features, headless)
                - info/
                    - analysis_config.json
            - model-quality/
                - baseline.csv
                - info/
                    - constraints.json
                    - statistics.json
                - reports/
                    - merge/
                        - XXX
```