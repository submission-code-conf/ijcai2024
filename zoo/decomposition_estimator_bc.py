# -*- coding: utf-8 -*-


import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
logging.getLogger().addHandler(logging.StreamHandler())
sys.path.append("/home/wyj/Iteraction_Modeling-develop_20230720")

from zoo.task_assembler import TaskAssembler
from zoo.feature.feature_process import FeatureProcess
import tensorflow_datasets as tfds

# tf.config.run_functions_eagerly(True)
# tf.enable_eager_execution()
# tf.executing_eagerly()

class DecompositionEstimator(tf.estimator.Estimator):
    def __init__(self,
                 tf_config,
                 params):
        # self.dataset = dataset

        def _model_fn(features, labels, mode, params):
            label_keys = 'not exists'
            if labels is not None:
                label_keys = labels.keys()
            print(str.format('model_fn_input, mode: {}, label.keys():≤ {}', mode, label_keys))

            multi_task_assembler = TaskAssembler(params)

            feature_process = FeatureProcess(params.get('feature_map_path'), params.get('default_embedding_dim', 8))

            input_embedding_list = feature_process.to_sparse_feature_embeddings(features)
            print('input_embedding_list', input_embedding_list)

            loss, output_dict, metric_dict = multi_task_assembler.assemble_model(input_embedding_list, None, labels,
                                                                                 features, mode)
            print('output_dict.keys(): {}, metric_dict.keys(): {}'.format(output_dict.keys(), metric_dict.keys()))

            if len(params.get('sample_id_cols', [])) > 0:
                for sample_id_col in params.get('sample_id_cols', []):
                    output_dict[sample_id_col] = tf.identity(features[sample_id_col])

            predictions = output_dict

            if mode == tf.estimator.ModeKeys.PREDICT:
                export_outputs = {
                    'prediction':
                        tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(
                    mode,
                    predictions=predictions,
                    export_outputs=export_outputs)

            # 训练
            if mode == tf.estimator.ModeKeys.TRAIN:
                for metric_name, metric_value in metric_dict.items():
                    loss = tf.compat.v1.Print(loss, [tf.shape(metric_value), metric_value], metric_name, summarize=10)

                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

                train = optimizer.minimize(
                    loss,
                    global_step=tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(
                    mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode,
                    predictions=predictions,
                    loss=loss,
                )

        super(DecompositionEstimator, self).__init__(
            model_fn=_model_fn,
            model_dir=tf_config.model_dir,
            config=tf_config,
            params=params)

# The original 'BX-Books' csv file had multiple cases of semi-colons in the book titles.
# To ensure proper dataframe importing, using Books.csv from Kaggle which Manually fixed instead.
# details: https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset
ratings_df = pd.read_csv('.././tensorflow_datasets/book-crossing/BX-Book-Ratings.csv', sep=';', encoding='latin1')
users_df = pd.read_csv('.././tensorflow_datasets/book-crossing/BX-Users.csv', sep=';', encoding='latin1')
books_df = pd.read_csv('.././tensorflow_datasets/book-crossing/Books.csv', sep=';', encoding='utf-8')

# 'User-ID', 'ISBN': int,
# 'Location': string,
# 'Title', 'Author': string
# 'Book-Rating: int
features = ['User-ID', 'Location', 'ISBN', 'Title', 'Author']
label = 'Book-Rating'
merged_df = pd.merge(pd.merge(ratings_df, books_df, on='ISBN'), users_df, on='User-ID')
merged_df = merged_df[features + [label]]

sz = len(merged_df)
merged_df = merged_df.sample(frac=1).reset_index(drop=True)
train_df = merged_df[0:int(0.8*sz)]
test_df = merged_df[int(0.8*sz):]
test_df = test_df[test_df['User-ID'].isin(train_df['User-ID'])]   # prevent the occurrence of new users in the test
test_df.to_csv("tzzs_data1_bc.csv")

print(len(merged_df))
print(len(train_df))
print(len(test_df))

def input_fn():
    features_dict = {}
    for feature in features:
        features_dict[feature] = train_df[feature].values.astype("str")
    features_dict[label] = train_df[label].values
    ratings = tf.data.Dataset.from_tensor_slices(features_dict)

    def _label_trans(y):
        return 1.0 if y >= 6 else 0.0

    ratings = ratings.map(
        lambda x: (
            {
                "User-ID": x["User-ID"],
                "Location": x["Location"],
                "ISBN": x["ISBN"],
                "Title": x["Title"],
                "Author": x["Author"],
                # "user_occupation_text": str(x[1]),
                # "user_zip_code": str(x[1])
            }
            , {
                "rating": x["Book-Rating"],
                "score_level": _label_trans(x["Book-Rating"])
            }
        ))
    # df = tfds.as_dataframe(ratings)
    # df.to_csv("tzzs_data_train_1.csv")

    shuffled = ratings.shuffle(1_000_000,
                               seed=2021,
                               reshuffle_each_iteration=False)
    # epochs
    dataset = shuffled.repeat(1)

    # batch size
    dataset = dataset.batch(1024)
    return dataset

def input_fn_test():
    features_dict = {}
    for feature in features:
        features_dict[feature] = test_df[feature].values.astype("str")
    features_dict[label] = test_df[label].values
    ratings = tf.data.Dataset.from_tensor_slices(features_dict)

    def _label_trans(y):
        return 1.0 if y >= 6 else 0.0

    ratings = ratings.map(
        lambda x: (
            {
                "User-ID": x["User-ID"],
                "Location": x["Location"],
                "ISBN": x["ISBN"],
                "Title": x["Title"],
                "Author": x["Author"],
                # "user_occupation_text": str(x[1]),
                # "user_zip_code": str(x[1])
            }
            , {
                "rating": x["Book-Rating"],
                "score_level": _label_trans(x["Book-Rating"])
            }
        ))
    # batch size
    dataset = ratings.batch(512)
    return dataset


if __name__ == '__main__':
    config_path = './config/d_c_2.json'
    params = json.load(open(config_path))
    params['sample_id_cols'] = ['User-ID', 'ISBN', 'Location', 'Title', 'Author']
    # print(params)

    model_config = tf.estimator.RunConfig(log_step_count_steps=100,
                                          save_summary_steps=100,
                                          save_checkpoints_steps=100,
                                          save_checkpoints_secs=None,
                                          keep_checkpoint_max=2)

    dataset = input_fn()
    estimator = DecompositionEstimator(model_config, params)

    def AUC(labels, predictions):
        print("predictions: ", predictions)
        print("labels: ", labels)
        auc_metric = tf.keras.metrics.AUC()
        # pred = tf.expand_dims(predictions['qi_score_logit'], axis=-1)
        print("E predictions:", predictions)
        pred = predictions['qi_score']
        print("D pred: ", pred)
        auc_metric.update_state(y_true=labels['score_level'], y_pred=pred)
        return {'auc': auc_metric}
    estimator = tf.compat.v1.estimator.add_metrics(estimator, AUC)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_test)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    predict_results = estimator.predict(input_fn=input_fn_test)
    predict_result_list = []
    for predict_result in predict_results:
        predict_result_list.append(predict_result)

    df = pd.DataFrame.from_records(predict_result_list)


    # run the graph
    # with tf.compat.v1.Session() as sess:
    #     # sess.run(init_op)  # execute init_op
    #     sess.run(df['user_id'])
    #     sess.run(df['movie_id'])

    print(df.head())
    df.to_csv("tzzs_data2_bc.csv")


