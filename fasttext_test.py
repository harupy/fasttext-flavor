import mlflow
import fasttext
import fasttext_flavor

# How to run this script:
# $ git git clone --single-branch --branch fasttext-flavor https://github.com/harupy/mlflow.git
# $ cd mlflow
# $ pip install fastext mlflow
# $ python fasstext


model = fasttext.train_supervised(input="cooking.train")

artifact_path = "model"
with mlflow.start_run() as run:
    fasttext_flavor.log_model(model, artifact_path)

model_uri = "runs:/{}/{}".format(run.info.run_id, artifact_path)
loaded_model = fasttext_flavor.load_model(model_uri)

pred = model.predict(
    "Which baking dish is best to bake a banana bread ?"
)

print(pred)
