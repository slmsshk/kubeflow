from kfp import dsl

@dsl.pipeline(
    name="kubeflow-demo-pipeline",
    description="A one-step demo pipeline"
)
def demo_pipeline():
    dsl.ContainerOp(
        name="say-hello",
        image="python:3.9-slim",
        command=["sh", "-c"],
        arguments=["echo 'Hello from Kubeflow demo!'"]
    )
