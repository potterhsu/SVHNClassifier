import tensorflow as tf
from model import Model

tf.app.flags.DEFINE_string('image', None, 'Path to image file')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
FLAGS = tf.app.flags.FLAGS


def main(_):
    path_to_image_file = FLAGS.image
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint

    image = tf.image.decode_jpeg(tf.read_file(path_to_image_file), channels=3)
    image = tf.reshape(image, [64, 64, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    image = tf.image.resize_images(image, [54, 54])
    images = tf.reshape(image, [1, 54, 54, 3])

    length_logits, digits_logits = Model.inference(images, drop_rate=0.0)
    length_predictions = tf.argmax(length_logits, axis=1)
    digits_predictions = tf.argmax(digits_logits, axis=2)
    digits_predictions_string = tf.reduce_join(tf.as_string(digits_predictions), axis=1)

    with tf.Session() as sess:
        restorer = tf.train.Saver()
        restorer.restore(sess, path_to_restore_checkpoint_file)

        length_predictions_val, digits_predictions_string_val = sess.run([length_predictions, digits_predictions_string])
        length_prediction_val = length_predictions_val[0]
        digits_prediction_string_val = digits_predictions_string_val[0]
        print 'length: %d' % length_prediction_val
        print 'digits: %s' % digits_prediction_string_val


if __name__ == '__main__':
    tf.app.run(main=main)