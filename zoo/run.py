import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from keras.layers import Dense

from zoo.feature.feature_process import to_sparse_feature_embeddings

flags.DEFINE_string('modelpython_dir', "./ckpt", 'export_dir')
flags.DEFINE_string('export_dir', "./export_dir", 'export_dir')
flags.DEFINE_string('model_dir', "./model_dir", 'model_dir')
flags.DEFINE_string('mode', "train", 'train or export')



FLAGS = flags.FLAGS


def input_fn():
    ratings = tfds.load("movielens/100k-ratings", split="train")
    ratings = ratings.map(
        lambda x: (
            {
                "movie_id": x["movie_id"],
                "user_id": x["user_id"],
            }, x["user_rating"]
        ))
    shuffled = ratings.shuffle(1_000_000,
                               seed=2021,
                               reshuffle_each_iteration=False)
    dataset = shuffled.batch(5)
    return dataset


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    sparse_feature_embeddings = to_sparse_feature_embeddings(features, ['user_id', 'movie_id'])

    x = tf.concat(sparse_feature_embeddings, axis=-1)
    hidden_units_list = [128, 32]
    for hidden_units in hidden_units_list:
        x = Dense(hidden_units, activation='relu')(x)

    logit = Dense(units=1)(x)
    prediction = logit

    loss = tf.keras.losses.MeanSquaredError()(labels, prediction)

    predictions = {"prediction": prediction}

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss, global_step=tf.compat.v1.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "predict_export_outputs":
                tf.estimator.export.PredictOutput(outputs=predictions)
        }
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)


def train(model_dir, ps_num):
    model_config = tf.estimator.RunConfig(log_step_count_steps=100,
                                          save_summary_steps=100,
                                          save_checkpoints_steps=100,
                                          save_checkpoints_secs=None,
                                          keep_checkpoint_max=2)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       params={"ps_num": ps_num},
                                       config=model_config)

    train_spec = tf.estimator.TrainSpec(input_fn=input_fn)

    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def serving_input_receiver_dense_fn():
    input_spec = {
        "movie_id": tf.constant([1], tf.int64),
        "user_id": tf.constant([1], tf.int64),
        "user_rating": tf.constant([1.0], tf.float32)
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(input_spec)


def export_for_serving(model_dir, export_dir, ps_num):
    model_config = tf.estimator.RunConfig(log_step_count_steps=100,
                                          save_summary_steps=100,
                                          save_checkpoints_steps=100,
                                          save_checkpoints_secs=None)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       params={"ps_num": ps_num},
                                       config=model_config)

    estimator.export_saved_model(export_dir, serving_input_receiver_dense_fn())


def main(argv):
    # del argv
    # tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    # task_name = tf_config.get('task', {}).get('type')
    # task_idx = tf_config.get('task', {}).get('index')
    #
    # ps_num = len(tf_config["cluster"]["ps"])

    if FLAGS.mode == "train":
        train(FLAGS.model_dir, 0)
    # if FLAGS.mode == "serving" and task_name == "chief" and int(task_idx) == 0:
    #     tfra.dynamic_embedding.enable_inference_mode()
    #     export_for_serving(FLAGS.model_dir, FLAGS.export_dir, ps_num)


if __name__ == "__main__":
    app.run(main)