"""
Based on the XGBoost flavor
"""
import os
import sys
import yaml
import logging
import pkg_resources

from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException


FLAVOR_NAME = "fasttext"
SERIALIZED_MODEL_FILE = "model.fasttext"

_logger = logging.getLogger(__name__)


def _get_installed_fasttext_version():
    # fasttext does not have a `__version__` attribute
    return pkg_resources.get_distribution("fasttext").version


def get_default_conda_env():
    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["fasttext=={}".format(_get_installed_fasttext_version())],
        additional_conda_channels=None,
    )


def save_model(
    ft_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
):
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    model_data_subpath = SERIALIZED_MODEL_FILE
    model_data_path = os.path.join(path, model_data_subpath)

    ft_model.save_model(model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module=__name__,
        data=model_data_subpath,
        env=conda_env_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME, fasttext_version=_get_installed_fasttext_version(), data=model_data_subpath
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(
    ft_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    **kwargs
):
    Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        registered_model_name=registered_model_name,
        ft_model=ft_model,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        **kwargs
    )


def _load_model(path):
    import fasttext
    return fasttext.load_model(path)


def _load_pyfunc(path):
    return _FastTextModelWrapper(_load_model(path))


def load_model(model_uri):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    model_file_path = os.path.join(
        local_model_path, flavor_conf.get("data", SERIALIZED_MODEL_FILE)
    )
    return _load_model(path=model_file_path)


class _FastTextModelWrapper:
    def __init__(self, ft_model):
        self.ft_model = ft_model

    def predict(self, dataframe):
        # Implement if necessary
        pass