import kfp
from kfp import dsl
from kfp.components import create_component_from_func

# Create Kubeflow components
preprocess_op = create_component_from_func('components/data_preprocessing.py')
train_op = create_component_from_func('components/train_model.py')
evaluate_op = create_component_from_func('components/evaluate_model.py')
save_op = create_component_from_func('components/save_model.py')
deploy_op = create_component_from_func('components/deploy_model.py')

@dsl.pipeline(
    name='Iris Versioned Pipeline',
    description='Kubeflow pipeline for Iris dataset with versioned models and conditional deployment'
)
def iris_pipeline():
    preprocess = preprocess_op()
    train = train_op().after(preprocess)
    evaluate = evaluate_op().after(train)
    save = save_op().after(evaluate)

    # Conditional deployment if accuracy > 0.9
    with dsl.Condition(
        dsl.load_component_from_file('components/evaluate_model.py')() > 0.9
    ):
        deploy_op(model_path=save.outputs['output'])

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(iris_pipeline, 'iris_pipeline.yaml')
