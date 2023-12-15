# -*- coding: utf-8 -*-

import json
import logging

import pandas as pd
import tensorflow as tf

logging.getLogger().addHandler(logging.StreamHandler())  # Set up logging handler

from task_assembler import TaskAssembler
from feature.feature_process import FeatureProcess
import tensorflow_datasets as tfds  # TensorFlow datasets library


class DecompositionEstimator(tf.estimator.Estimator):
    def __init__(self, tf_config, params):
        """
        Initialization of DecompositionEstimator, a custom TensorFlow Estimator.
        """

        def _model_fn(features, labels, mode, params):
            """
            The model function for the Estimator, defining the model's behavior.
            """
            label_keys = 'not exists'
            if labels is not None:
                label_keys = labels.keys()

            # Logging inputs for debugging
            print(str.format('model_fn_input, mode: {}, label.keys(): {}', mode, label_keys))

            multi_task_assembler = TaskAssembler(params)
            feature_process = FeatureProcess(params.get('feature_map_path'), params.get('default_embedding_dim', 8))

            # Processing input features into embeddings
            input_embedding_list = feature_process.to_sparse_feature_embeddings(features)
            print('input_embedding_list', input_embedding_list)

            # Assembling the model and calculating loss and metrics
            loss, output_dict, metric_dict = multi_task_assembler.assemble_model(input_embedding_list, None, labels,
                                                                                 features, mode)
            print('output_dict.keys(): {}, metric_dict.keys(): {}'.format(output_dict.keys(), metric_dict.keys()))

            # Adding sample ID columns to output if specified
            if len(params.get('sample_id_cols', [])) > 0:
                for sample_id_col in params.get('sample_id_cols', []):
                    output_dict[sample_id_col] = tf.identity(features[sample_id_col])

            predictions = output_dict

            # Setting up EstimatorSpec for PREDICT mode
            if mode == tf.estimator.ModeKeys.PREDICT:
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            # Setting up EstimatorSpec for TRAIN mode
            if mode == tf.estimator.ModeKeys.TRAIN:
                # Adding metrics to logging
                for metric_name, metric_value in metric_dict.items():
                    loss = tf.compat.v1.Print(loss, [tf.shape(metric_value), metric_value], metric_name, summarize=10)

                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
                train = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, loss=loss, train_op=train)

            # Setting up EstimatorSpec for EVAL mode
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, loss=loss)

        # Initializing the parent class
        super(DecompositionEstimator, self).__init__(model_fn=_model_fn, model_dir=tf_config.model_dir,
                                                     config=tf_config, params=params)


def input_fn():
    """
    Input function for training data.
    """
    # Loading and preprocessing the MovieLens dataset
    ratings = tfds.load("movielens/100k-ratings", split="train[:80%]")

    # or loading and preprocessing the MovieLens dataset
    # ratings = tfds.load("movielens/1m-ratings", split="train[:80%]")

    def _label_trans(y):
        # Transforming the label
        return 1.0 if y >= 3 else 0.0

    ratings = ratings.map(lambda x: (
    {"movie_id": x["movie_id"], "movie_title": x["movie_title"], "user_id": x["user_id"],
     "user_occupation_text": x["user_occupation_text"], "user_zip_code": x["user_zip_code"]},
    {"user_rating": x["user_rating"], "score_level": _label_trans(x["user_rating"])}))

    # Shuffling and batching the dataset
    shuffled = ratings.shuffle(1_000_000, seed=2021, reshuffle_each_iteration=False)
    dataset = shuffled.repeat(10)  # Setting the number of epochs
    dataset = dataset.batch(512)  # Setting batch size
    return dataset


def input_fn_test():
    """
    Input function for testing data.
    """
    # Loading the test portion of the MovieLens dataset
    ratings = tfds.load("movielens/100k-ratings", split="train[80%:100%]")
    df = tfds.as_dataframe(ratings)
    df.to_csv("tzzs_data1.csv")  # Saving the dataset to a CSV file

    def _label_trans(y):
        # Transforming the label
        return 1.0 if y >= 3 else 0.0

    print("ratings: ", ratings)
    ratings = ratings.map(lambda x: (
    {"movie_id": x["movie_id"], "movie_title": x["movie_title"], "user_id": x["user_id"],
     "user_occupation_text": x["user_occupation_text"], "user_zip_code": x["user_zip_code"]},
    {"user_rating": x["user_rating"], "score_level": _label_trans(x["user_rating"])}))
    dataset = ratings.batch(512)  # Batching the test dataset
    return dataset


if __name__ == '__main__':
    # Main execution block
    config_path = 'decompose_config.json'
    params = json.load(open(config_path))
    params['sample_id_cols'] = ['user_id', 'movie_id', 'user_occupation_text', 'user_zip_code', 'movie_title']
    print(params)

    # Configuring the model
    model_config = tf.estimator.RunConfig(log_step_count_steps=100, save_summary_steps=100, save_checkpoints_steps=100,
                                          save_checkpoints_secs=None, keep_checkpoint_max=2)

    dataset = input_fn()  # Getting the training dataset
    estimator = DecompositionEstimator(model_config, params)  # Initializing the Estimator


    def AUC(labels, predictions):
        """
        A function to compute the AUC metric.
        """
        print("predictions: ", predictions)
        print("labels: ", labels)
        auc_metric = tf.keras.metrics.AUC()
        pred = predictions['qi_score']
        auc_metric.update_state(y_true=labels['score_level'], y_pred=pred)
        return {'auc': auc_metric}


    estimator = tf.compat.v1.estimator.add_metrics(estimator, AUC)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # Setting up training and evaluation specifications
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_test)

    # Training and evaluating the model
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Making predictions with the trained model
    predict_results = estimator.predict(input_fn=input_fn_test)
    predict_result_list = []
    for predict_result in predict_results:
        predict_result_list.append(predict_result)

    # Storing predictions in a DataFrame and saving to CSV
    df = pd.DataFrame.from_records(predict_result_list)
    print(df.head())
    df.to_csv("demo_data.csv")
