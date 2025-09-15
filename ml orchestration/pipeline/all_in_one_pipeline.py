from pathlib import Path
from kfp import dsl

# All steps share the same Docker image that already contains
# data_preprocessing.py, train_model.py, evaluate_model.py,
# save_model.py, deploy_model.py, plus all dependencies.
BASE_IMAGE = "iris-all:latest"
COMP_DIR   = "/app/components"

# ---------- Components ----------
@dsl.component(base_image=BASE_IMAGE)
def preprocess_op():
    import subprocess
    subprocess.run(["python", f"{COMP_DIR}/data_preprocessing.py"], check=True)

@dsl.component(base_image=BASE_IMAGE)
def train_op():
    import subprocess
    subprocess.run(["python", f"{COMP_DIR}/train_model.py"], check=True)

@dsl.component(base_image=BASE_IMAGE)
def evaluate_op() -> float:
    import subprocess, pathlib
    # The evaluate_model.py script must write /tmp/accuracy.txt
    subprocess.run(["python", f"{COMP_DIR}/evaluate_model.py"], check=True)
    acc = pathlib.Path("/tmp/accuracy.txt").read_text().strip()
    return float(acc)

@dsl.component(base_image=BASE_IMAGE)
def save_op():
    import subprocess
    subprocess.run(["python", f"{COMP_DIR}/save_model.py"], check=True)

@dsl.component(base_image=BASE_IMAGE)
def deploy_op():
    import subprocess
    subprocess.run(["python", f"{COMP_DIR}/deploy_model.py"], check=True)

# ---------- Pipeline ----------
@dsl.pipeline(name="Iris All-in-One Pipeline v2",
              description="Single Docker image, modern KFP v2 syntax")
def iris_all_pipeline():
    p = preprocess_op()
    t = train_op().after(p)
    e = evaluate_op().after(t)
    s = save_op().after(e)

    with dsl.Condition(e.output > 0.9):
        deploy_op().after(s)

# ---------- Compile ----------
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        iris_all_pipeline,
        package_path=str(Path(__file__).with_suffix(".yaml"))
    )
