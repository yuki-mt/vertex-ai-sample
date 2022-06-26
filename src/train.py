from kfp.v2.dsl import component, Dataset, Input, Metrics, Model, Output


@component(
    packages_to_install=["sklearn", "pandas", "joblib"],
    base_image="python:3.9",
    # output_component_file="train_model_component.yaml",
)
def sklearn_train(
    dataset: Input[Dataset],
    metrics: Output[Metrics],
    model: Output[Model]
):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from joblib import dump
    import pandas as pd

    df = pd.read_csv(dataset.path)
    labels = df.pop("target").tolist()
    data = df.values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(data, labels)

    skmodel = DecisionTreeClassifier()
    skmodel.fit(x_train, y_train)
    score = skmodel.score(x_test, y_test)

    metrics.log_metric("accuracy", score * 100)
    metrics.log_metric("framework", "Scikit Learn")
    metrics.log_metric("dataset_size", len(df))
    dump(skmodel, model.path + ".joblib")
