from pathlib import Path
from kfp import dsl

# ---------- Components ----------
@dsl.component(base_image="iris-preprocess:latest")
def preprocess_op():
    import subprocess
    subprocess.run(["python", "/app/data_preprocessing.py"], check=True)

@dsl.component(base_image="iris-train:latest")
def train_op():
    import subprocess
    subprocess.run(["python", "/app/train_model.py"], check=True)

@dsl.component(base_image="iris-evaluate:latest")
def evaluate_op() -> float:
    import subprocess, pathlib
    subprocess.run(["python", "/app/evaluate_model.py"], check=True)
    acc = pathlib.Path("/tmp/accuracy.txt").read_text().strip()
    return float(acc)

@dsl.component(base_image="iris-save:latest")
def save_op():
    import subprocess
    subprocess.run(["python", "/app/save_model.py"], check=True)

@dsl.component(base_image="iris-deploy:latest")
def deploy_op():
    import subprocess
    subprocess.run(["python", "/app/deploy_model.py"], check=True)

# ---------- Pipeline ----------
@dsl.pipeline(
    name="Iris Multi-Image Pipeline v2",
    description="Each component runs in its own custom Docker image (using dsl.If)."
)
def iris_multi_pipeline():
    p = preprocess_op()
    t = train_op().after(p)
    e = evaluate_op().after(t)
    s = save_op().after(e)

    with dsl.If(e.output > 0.9):
        deploy_op().after(s)

# ---------- Compile ----------
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        iris_multi_pipeline,
        package_path=str(Path(__file__).with_suffix(".yaml"))
    )
