"""ImageNet input pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from resnet.data_tfrecord.data_factory import RegisterInputPipeline
from resnet.data_tfrecord.input_pipeline import InputPipeline

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 224,
                            """Provide square images of this size.""")


@RegisterInputPipeline("imagenet")
class ImagenetInputPipeline(InputPipeline):
  """ImageNet data set input pipeline."""

  def parse_example_proto(self, example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature(
            [1], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update({
        k: sparse_float32
        for k in [
            'image/object/bbox/xmin', 'image/object/bbox/ymin',
            'image/object/bbox/xmax', 'image/object/bbox/ymax'
        ]
    })

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return {
        'image_buffer': features['image/encoded'],
        'label': label,
        'bbox': bbox,
        'text': features['image/class/text']
    }

  def decode_jpeg(self, image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(
        values=[image_buffer], name=scope, default_name='decode_jpeg'):
      # Decode the string as an RGB JPEG.
      # Note that the resulting image contains an unknown height and width
      # that is set dynamically by decode_jpeg. In other words, the height
      # and width of image is unknown at compile-time.
      image = tf.image.decode_jpeg(image_buffer, channels=3)

      # After this point, all image pixels reside in [0,1)
      # until the very end, when they're rescaled to (-1, 1).  The various
      # adjust_* ops all require this range for dtype float.
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return image

  def distort_color(self, image, thread_id=0, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for name_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(
        values=[image], name=scope, default_name='distort_color'):
      color_ordering = thread_id % 2

      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

      # The random_* ops do not necessarily clamp.
      image = tf.clip_by_value(image, 0.0, 1.0)
      return image

  def distort_image(self, image, height, width, bbox, thread_id=0, scope=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
      image: 3-D float Tensor of image
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      thread_id: integer indicating the preprocessing thread.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    with tf.name_scope(
        values=[image, height, width, bbox],
        name=scope,
        default_name='distort_image'):
      # Each bounding box has shape [1, num_boxes, box coords] and
      # the coordinates are ordered [ymin, xmin, ymax, xmax].
      # A large fraction of image datasets contain a human-annotated bounding
      # box delineating the region of the image containing the object of interest.
      # We choose to create a new bounding box for the object which is a randomly
      # distorted version of the human-annotated bounding box that obeys an allowed
      # range of aspect ratios, sizes and overlap with the human-annotated
      # bounding box. If no box is supplied, then we assume the bounding box is
      # the entire image.
      sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bbox,
          min_object_covered=0.1,
          aspect_ratio_range=[0.75, 1.33],
          area_range=[0.05, 1.0],
          max_attempts=100,
          use_image_if_no_bounding_boxes=True)
      bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
      if not thread_id:
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distort_bbox)
        tf.summary.image('images_with_distorted_bounding_box',
                         image_with_distorted_box)

      # Crop the image to the specified bounding box.
      distorted_image = tf.slice(image, bbox_begin, bbox_size)

      # This resizing operation may distort the images because the aspect
      # ratio is not respected. We select a resize method in a round robin
      # fashion based on the thread number.
      # Note that ResizeMethod contains 4 enumerated resizing methods.
      resize_method = thread_id % 4
      distorted_image = tf.image.resize_images(
          distorted_image, [height, width], method=resize_method)
      # Restore the shape since the dynamic slice based upon the bbox_size loses
      # the third dimension.
      distorted_image.set_shape([height, width, 3])

      # Randomly flip the image horizontally.
      distorted_image = tf.image.random_flip_left_right(distorted_image)

      # Randomly distort the colors.
      distorted_image = self.distort_color(distorted_image, thread_id)
      return distorted_image

  def eval_image(self, image, height, width, scope=None):
    """Prepare one image for evaluation.

    Args:
      image: 3-D float Tensor
      height: integer
      width: integer
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(
        values=[image, height, width], name=scope, default_name='eval_image'):
      # Crop the central region of the image with an area containing 87.5% of
      # the original image.
      image = tf.image.central_crop(image, central_fraction=0.875)

      # Resize the image to the original height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(
          image, [height, width], align_corners=False)
      image = tf.squeeze(image, [0])
      return image

  def preprocess_example(self, example, is_training, thread_id=0):
    """Decode and preprocess one image for evaluation or training.

    Args:
      example: Dictionary contains a data entry.
      train: boolean
      thread_id: integer indicating preprocessing thread

    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
      ValueError: if user does not provide bounding box
    """
    image_buffer = example['image_buffer']
    bbox = example['bbox']
    if bbox is None:
      raise ValueError('Please supply a bounding box.')

    image = self.decode_jpeg(image_buffer)
    height = FLAGS.image_size
    width = FLAGS.image_size

    if is_training:
      image = self.distort_image(image, height, width, bbox, thread_id)
    else:
      image = self.eval_image(image, height, width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    # Make label 0-based.
    label = tf.subtract(example['label'], 1)
    return {'image': image, 'label': label}
