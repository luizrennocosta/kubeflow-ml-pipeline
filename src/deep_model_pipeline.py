import kfp.dsl as dsl


@dsl.pipeline(
  name='NLP',
  description='A pipeline demonstrating reproducible steps for NLP'
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
      name='my-pvc',
      resource_name="my-pvc",
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

    download_embed_step = dsl.ContainerOp(
        name='embed_weights_downloader',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/weights_downloader/pipeline_step.py",
            "--url", embed_weights_url
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

    tokenize_step = dsl.ContainerOp(
        name='tokenize',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/tokenize/pipeline_step.py",
            "--in-path", "/mnt/data/train.data",
            "--out-path", tokens_path,
            "--word-index-path", word_index_path,
            "--tokenizer-path", "/mnt/tokenizer.model",
            "--action", "train",
            "--num-words", num_words,
            "--max-length", sentence_max_length,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(data_split_step)

    tokenize_step_val = dsl.ContainerOp(
        name='val_tokenize',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/tokenize/pipeline_step.py",
            "--in-path", "/mnt/data/val.data",
            "--out-path", "/mnt/data/tokenized_val.data",
            "--word-index-path", word_index_path,
            "--tokenizer-path", "/mnt/tokenizer.model",
            "--action", "predict",
            "--num-words", num_words,
            "--max-length", sentence_max_length,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(tokenize_step)

    tokenize_step_test = dsl.ContainerOp(
        name='test_tokenize',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/tokenize/pipeline_step.py",
            "--in-path", "/mnt/data/test.data",
            "--out-path", "/mnt/data/tokenized_test.data",
            "--word-index-path", word_index_path,
            "--tokenizer-path", "/mnt/tokenizer.model",
            "--action", "predict",
            "--num-words", num_words,
            "--max-length", sentence_max_length,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(tokenize_step)

    embedding_step = dsl.ContainerOp(
        name='embedder',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/embedder/pipeline_step.py",
            "--in-path", word_index_path,
            "--out-path", embedded_matrix_path,
            "--glove-file", pre_embedded_weights,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(tokenize_step, download_embed_step)

    train_step = dsl.ContainerOp(
        name='predictor',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/deep_classifier/pipeline_step.py",
            "--in-path", tokens_path,
            "--embed-weight", embedded_matrix_path,
            "--out-path", model_prediction_path,
            "--action", "train",
            "--model-path", deep_model,
            "--epochs", 20,
            "--batch-size", 1024,
            "--optimizer", "rmsprop",
            "--metrics", ["acc"]
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(embedding_step)

    predict_val = dsl.ContainerOp(
        name='val_predictor',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/deep_classifier/pipeline_step.py",
            "--in-path", "/mnt/data/tokenized_val.data",
            "--embed-weight", embedded_matrix_path,
            "--out-path", '/mnt/predicted_val.data',
            "--action", "predict",
            "--model-path", deep_model,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(train_step, tokenize_step_val)

    predict_test = dsl.ContainerOp(
        name='test_predictor',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/deep_classifier/pipeline_step.py",
            "--in-path", "/mnt/data/tokenized_test.data",
            "--embed-weight", embedded_matrix_path,
            "--out-path", '/mnt/predicted_test.data',
            "--action", "predict",
            "--model-path", deep_model,
        ],
        pvolumes={"/mnt": vop.volume}
    ).after(train_step, tokenize_step_test)

    evaluate_model = dsl.ContainerOp(
        name='model_evaluator',
        image='docker.io/cyferino/component-kubeflow:0.0.14',
        command="python",
        arguments=[
            "/src/pipeline_steps/deep_classifier/pipeline_step.py",
            "--data-folder", "/mnt/data",
            "--predicted-train-data", '/mnt/predicted_train.data',
            "--predicted-val-data", '/mnt/predicted_val.data',
            "--predicted-test-data", '/mnt/predicted_test.data'
        ],
        pvolumes={"/mnt": vop.volume},
        file_outputs={
            'mlpipeline-metrics': '/mlpipeline-metrics.json',
            'mlpipeline-ui-data': '/mlpipeline-ui-data.json'
        },
        output_artifact_paths={
            'mlpipeline-metrics': '/mlpipeline-metrics.json',
            'mlpipeline-ui-data': '/mlpipeline-ui-data.json'
        }
    ).after(train_step, predict_val, predict_test)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(nlp_pipeline, __file__ + '.tar.gz')