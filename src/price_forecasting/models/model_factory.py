from price_forecasting.models import MODEL_REGISTRY

def build_model(config: dict):

    if "model" not in config:
        model_name = "QuantileLSTM" #backwards compatibility for first model
    else:
        model_name = config["model"]
    model_class = MODEL_REGISTRY.get(model_name)

    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    return model_class(**config)