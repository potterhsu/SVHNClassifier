import os
import tensorflow as tf
from meta import Meta
from evaluator import Evaluator

tf.app.flags.DEFINE_string('data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('checkpoint_dir', './logs/train', 'Directory to read checkpoint files')
tf.app.flags.DEFINE_string('eval_logdir', './logs/eval', 'Directory to write evaluation logs')
FLAGS = tf.app.flags.FLAGS


def _eval(path_to_checkpoint_dir, path_to_eval_tfrecords_file, num_eval_examples, path_to_eval_log_dir):
    evaluator = Evaluator(path_to_eval_log_dir)

    checkpoint_paths = tf.train.get_checkpoint_state(path_to_checkpoint_dir).all_model_checkpoint_paths
    for global_step, path_to_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
        try:
            global_step_val = int(global_step)
        except ValueError:
            continue

        accuracy = evaluator.evaluate(path_to_checkpoint, path_to_eval_tfrecords_file, num_eval_examples,
                                      global_step_val)
        print 'Evaluate %s on %s, accuracy = %f' % (path_to_checkpoint, path_to_eval_tfrecords_file, accuracy)


def main(_):
    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
    path_to_checkpoint_dir = FLAGS.checkpoint_dir

    path_to_train_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'train')
    path_to_val_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'val')
    path_to_test_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'test')

    meta = Meta()
    meta.load(path_to_tfrecords_meta_file)

    _eval(path_to_checkpoint_dir, path_to_train_tfrecords_file, meta.num_train_examples, path_to_train_eval_log_dir)
    _eval(path_to_checkpoint_dir, path_to_val_tfrecords_file, meta.num_val_examples, path_to_val_eval_log_dir)
    _eval(path_to_checkpoint_dir, path_to_test_tfrecords_file, meta.num_test_examples, path_to_test_eval_log_dir)


if __name__ == '__main__':
    tf.app.run(main=main)
