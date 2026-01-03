import mlflow


def start_mlflow(baseline,uri):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("GroupActivityRecognition")
    mlflow.start_run(run_name=baseline)

def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)



def log_metrics(metrics, step=None):
    if step:
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)
        return
    
    for k, v in metrics.items():
            mlflow.log_metric(k, v)


def end_mlflow():
    mlflow.end_run()
