from kfp.v2.dsl import component, Dataset, OutputPath


@component(
    packages_to_install=["sklearn", "pandas"],
    base_image="python:3.9",
)
def get_dataframe(output_data_path: OutputPath(Dataset)):  # type: ignore
    import pandas as pd
    from sklearn import datasets

    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target_names[iris.target]
    df.to_csv(output_data_path, index=False)
