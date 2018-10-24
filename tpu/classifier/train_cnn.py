import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os


# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing the dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 1024,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS





 def load_np_data(path):
     data = []
     labels = []
     images_files = list(filter(lambda x: x.endswith('npy'), os.listdir(path)))
     label_names = [x.split('.')[0] for x in images_files]
     class_mapping = {}
     for i in range(len(images_files)):
         class_mapping[i] = label_names[i]
         print(os.path.join(path, images_files[i]))
         samples = np.load(os.path.join(path, images_files[i]))[:4000]
         data.append(samples)
         labels.append(np.array([i] * len(samples)))

     return (np.concatenate(data), np.concatenate(labels)), class_mapping

 def get_model(features, mode):
     """Model function for CNN."""
     # Input Layer
     input_layer = tf.reshape(features, [-1, 28, 28, 1])

     # Convolutional Layer #1
     conv1 = tf.layers.conv2d(
         inputs=input_layer,
         filters=32,
         kernel_size=[5, 5],
         padding="same",
         activation=tf.nn.relu)

     # Pooling Layer #1
     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

     # Convolutional Layer #2 and Pooling Layer #2
     conv2 = tf.layers.conv2d(
         inputs=pool1,
         filters=64,
         kernel_size=[5, 5],
         padding="same",
         activation=tf.nn.relu)
     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

     # Dense Layer
     pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
     dropout = tf.layers.dropout(
         inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

     # Logits Layer
     logits = tf.layers.dense(inputs=dropout, units=10)

     return logits

 def metric_fn(labels, logits):
   accuracy = tf.metrics.accuracy(
       labels=labels, predictions=tf.argmax(logits, axis=1))
   return {"accuracy": accuracy}

 def cnn_model_fn(features, labels, mode, params):

     logits = get_model(features, mode)
     predictions = {
         # Generate predictions (for PREDICT and EVAL mode)
         "classes": tf.argmax(input=logits, axis=1),
         # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
         # `logging_hook`.
         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
     }

     if mode == tf.estimator.ModeKeys.PREDICT:
         return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

     # Calculate Loss (for both TRAIN and EVAL modes)
     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

     # Configure the Training Op (for TRAIN mode)
     if mode == tf.estimator.ModeKeys.TRAIN:
         optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
         optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
         train_op = optimizer.minimize(
             loss=loss,
             global_step=tf.train.get_global_step())
         return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op)

     if mode == tf.estimator.ModeKeys.EVAL:
         return tf.contrib.tpu.TPUEstimatorSpec(
             mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


 def train_input_fn(params):
     """train_input_fn defines the input pipeline used for training."""
     batch_size = params["batch_size"]
     data_dir = params["data_dir"]

     (X, y), mappings = load_np_data(data_dir)
     X = X.astype(np.float32) / 255.0
     y = y.astype(np.int32)
     dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size, drop_remainder=True).repeat().shuffle(
         buffer_size=50000)

     features, labels = dataset.make_one_shot_iterator().get_next()
     return features, labels

 def eval_input_fn(params):
     """train_input_fn defines the input pipeline used for training."""
     batch_size = params["batch_size"]
     data_dir = params["data_dir"]

     (X, y), mappings = load_np_data(data_dir)
     X = X.astype(np.float32) / 255.0
     y = y.astype(np.int32)
     dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size, drop_remainder=True)\
         .repeat().shuffle(buffer_size=50000)

     features, labels = dataset.make_one_shot_iterator().get_next()
     return features, labels


 def main(argv):

     del argv

     tf.logging.set_verbosity(tf.logging.INFO)

     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
         FLAGS.tpu,
         zone=FLAGS.tpu_zone,
         project=FLAGS.gcp_project
     )

     run_config = tf.contrib.tpu.RunConfig(
         cluster=tpu_cluster_resolver,
         model_dir=FLAGS.model_dir,
         session_config=tf.ConfigProto(
             allow_soft_placement=True, log_device_placement=True),
         tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
     )

     # Create the Estimator
     classifier = tf.contrib.tpu.TPUEstimator(
         model_fn=cnn_model_fn,
         use_tpu=FLAGS.use_tpu,
         train_batch_size=FLAGS.batch_size,
         eval_batch_size=FLAGS.batch_size,
         predict_batch_size=FLAGS.batch_size,
         config=run_config,
         params={"data_dir": FLAGS.data_dir}
     )

     classifier.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

     eval_results = classifier.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)
     print(eval_results)


if __name__ == '__main__':
    tf.app.run()