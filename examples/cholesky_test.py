import tensorflow as tf

from deep_sort.sort.kalman_filter import KalmanFilter


def split_and_cat(means, covariances):
    r_means = []
    r_covs = []
    for i in range(len(means)):
        r_means.append(means[i].unsqueeze(0))
        r_covs.append(covariances[i].unsqueeze(0))

    return tf.concat(r_means, 0), tf.concat(r_covs, 0)


def gen_means_and_covs(size=15):
    means = []
    covariances = []
    for i in range(size):
        mean, covariance = kf.initiate(tf.random.uniform([4]))
        means.append(mean)
        covariances.append(covariance)

    means = tf.concat(means, 0)
    covariances = tf.concat(covariances,0)

    return means, covariances


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    kf = KalmanFilter()

    means = []
    covariances = []
    measures = []
    for i in range(15):
        mean, covariance = kf.initiate(tf.random.uniform([4]))
        means.append(mean)
        covariances.append(covariance)
        measures.append(tf.random.uniform([4]))

    means = tf.concat(means, 0)
    covariances = tf.concat(covariances, 0)
    measures = tf.stack(measures, 0)

    for i in range(10):
        print(i)

        means, covariances = gen_means_and_covs(15)

        means, covariances = kf.predict(means, covariances)

        squared_maha = kf.gating_distance(means,
                                          covariances,
                                          tf.constant([[12, 20, 0.6, 11],
                                                        [20, 16, 0.4, 18],
                                                        [20, 16, 0.4, 18],
                                                        [20, 16, 0.4, 18],
                                                        [20, 16, 0.4, 18],
                                                        [20, 16, 0.4, 18],
                                                        [20, 16, 0.4, 18],
                                                        [20, 16, 0.4, 18],
                                                        [20, 16, 0.4, 18],
                                                        ]))

        means, covariances = kf.update(means, covariances, measures)
