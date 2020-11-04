import kfp.dsl as dsl
import yaml


@dsl.pipeline(
  name='NLP',
  description='A pipeline demonstrating reproducible steps for NLP using logistic regression'
)
def nlp_pipeline(
        csv_url="https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_UK_v1_00.tsv.gz",
        embed_weights_url="http://nlp.stanford.edu/data/glove.42B.300d.zip",
        features_column="review_body",
        labels_column="product_category",
        raw_text_path='/mnt/text.data',
        labels_path='/mnt/labels.data',
        data_folder='/mnt/data',
        clean_text_path='/mnt/clean.data',
        tokens_path='/mnt/tokens.data',
        tfidf_vectors_path='/mnt/tfidf.data',
        model_prediction_path='/mnt/predicted_train.data',
        tfidf_model_path='/mnt/tfidf.model',
        word_index_path='/mnt/word_index.data',
        embedded_matrix_path='/mnt/embedded_matrix.data',
        pre_embedded_weights='/mnt/data/glove.42B.300d.txt',
        train_ratio=0.98,
        validation_ratio=0.01,
        test_ratio=0.01,
        num_words=20000,
        sentence_max_length=50,
        deep_model='/mnt/deep_model.model',
        batch_size='100'):
    """
    Pipeline
    """
    vop = dsl.VolumeOp(
      name='lr-pvc',
      resource_name="lr-pvc",
      modes=["ReadWriteMany"],
      size="30Gi"
    )

    download_step = dsl.ContainerOp(
        name='data_downloader',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/data_downloader/pipeline_step.py",
            "--labels-path", labels_path,
            "--data-folder", data_folder,
            "--features-path", raw_text_path,
            "--csv-url", csv_url,
            "--features-column", features_column,
            "--labels-column", labels_column
        ],
        pvolumes={"/mnt": vop.volume}
    )

    clean_step = dsl.ContainerOp(
        name='clean_text',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/clean_text/pipeline_step.py",
            "--in-path", raw_text_path,
            "--out-path", clean_text_path,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(download_step)

    data_split_step = dsl.ContainerOp(
        name='data_splitter',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/train_val_test/pipeline_step.py",
            "--in-path", clean_text_path,
            "--labels-path", labels_path,
            "--out-folder", data_folder,
            "--train-ratio", train_ratio,
            "--validation-ratio", validation_ratio,
            "--test-ratio", test_ratio
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(clean_step)

    tfidf_step = dsl.ContainerOp(
        name='tfidf',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/tfidf_vectorizer/pipeline_step.py",
            "--in-path", "/mnt/data/train.data",
            "--out-path", tokens_path,
            "--model-path", "/mnt/tfidf.model",
            "--action", "train",
            "--ngram-range", 2,
            "--max-features", 1000,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(data_split_step)

    tfidf_step_val = dsl.ContainerOp(
        name='tfidf_val',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/tfidf_vectorizer/pipeline_step.py",
            "--in-path", "/mnt/data/val.data",
            "--out-path", "/mnt/data/tokenized_val.data",
            "--model-path", "/mnt/tfidf.model",
            "--action", "predict",
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(tfidf_step)

    tfidf_step_test = dsl.ContainerOp(
        name='tfidf_test',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/tfidf_vectorizer/pipeline_step.py",
            "--in-path", "/mnt/data/test.data",
            "--out-path", "/mnt/data/tokenized_test.data",
            "--model-path", "/mnt/tfidf.model",
            "--action", "predict",
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(tfidf_step)

    train_step = dsl.ContainerOp(
        name='predictor',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/lr_text_model/pipeline_step.py",
            "--in-path", tokens_path,
            "--out-path", model_prediction_path,
            "--model-path", '/mnt/lr_text.model',
            "--action", "train",
            "--c-param", 0.1,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(tfidf_step)

    predict_val = dsl.ContainerOp(
        name='val_predictor',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/lr_text_model/pipeline_step.py",
            "--in-path", "/mnt/data/tokenized_val.data",
            "--out-path", '/mnt/predicted_val.data',
            "--model-path", '/mnt/lr_text.model',
            "--action", "predict",
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(train_step, tfidf_step_val)

    predict_test = dsl.ContainerOp(
        name='test_predictor',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/lr_text_model/pipeline_step.py",
            "--in-path", "/mnt/data/tokenized_test.data",
            "--out-path", '/mnt/predicted_test.data',
            "--model-path", '/mnt/lr_text.model',
            "--action", "predict",
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(train_step, tfidf_step_test)

    evaluate_model = dsl.ContainerOp(
        name='model_evaluator',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/evaluate_model/pipeline_step.py",
            "--data-folder", "/mnt/data",
            "--predicted-train-data", '/mnt/predicted_train.data',
            "--predicted-val-data", '/mnt/predicted_val.data',
            "--predicted-test-data", '/mnt/predicted_test.data'
        ],
        pvolumes={"/mnt": vop.volume},
        file_outputs={'mlpipeline-metrics': '/mlpipeline-metrics.json',
            'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'},
        output_artifact_paths={'mlpipeline-metrics': '/mlpipeline-metrics.json',
            'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    ).after(train_step, predict_val, predict_test)

    seldon_config = yaml.load(open("seldon_production_pipeline.yaml"))

    deploy_step = dsl.ResourceOp(
        name="seldondeploy",
        k8s_resource=seldon_config,
        attribute_outputs={"name": "{.metadata.name}"})

    deploy_step.after(train_step)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(nlp_pipeline, __file__ + '_lr.tar.gz')