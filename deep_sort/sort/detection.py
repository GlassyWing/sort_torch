import tensorflow as tf

class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    payload: any
        A param maintain something
    """

    def __init__(self, tlwh, confidence, feature, payload=None):
        self.tlwh = tf.cast(tlwh,tf.float32)
        self.confidence = float(confidence)
        self.feature = feature
        self.payload = payload

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = tf.identity(self.tlwh)
        return tf.concat((ret[:2], ret[2:] + ret[:2]), 0)

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = tf.identity(self.tlwh)
        return tf.concat((ret[0] + ret[2] / 2, ret[1] + ret[3] / 2, ret[2] / ret[3], ret[3]), 0)
