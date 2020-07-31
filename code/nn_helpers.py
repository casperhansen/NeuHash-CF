import tensorflow as tf

def extract_fn(data_record):
    '''
    'user': tf.train.Feature(int64_list = tf.train.Int64List(value = [sample[0]])),
    'i1': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[1]])),
    'i2': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[2]])),
    'i_unrated': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[3]])),
    'i1_rating': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[4]])),
    'i2_rating': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[5]]))
    '''

    features = {
        'user': tf.FixedLenFeature([1], tf.int64),
        'i1': tf.FixedLenFeature([1], tf.int64),
        'i2': tf.FixedLenFeature([1], tf.int64),
        'i_unrated': tf.FixedLenFeature([1], tf.int64),
        'i1_rating': tf.FixedLenFeature([1], tf.int64),
        'i2_rating': tf.FixedLenFeature([1], tf.int64),
    }

    sample = tf.parse_single_example(data_record, features)

    for key in features.keys():
        sample[key] = tf.squeeze(sample[key], -1)

    for key in ["user","i1","i2","i_unrated"]:
        sample[key] = tf.cast(sample[key], tf.int32)

    sample["i1_rating"] = tf.cast(sample["i1_rating"], tf.float32)
    sample["i2_rating"] = tf.cast(sample["i2_rating"], tf.float32)

    return tuple([sample[key] for key in ["user", "i1", "i2", "i_unrated", "i1_rating", "i2_rating"]])

def generator(sess, handle, batchsize, record_paths, is_test):

    output_t = tuple([tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32])
    s = tf.TensorShape([None,])
    output_s = tuple([s for _ in output_t])

    dataset = tf.data.Dataset.from_tensor_slices(record_paths)
    if not is_test:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(100)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(extract_fn, num_parallel_calls=3)
    if not is_test:
        dataset = dataset.shuffle(30000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_initializable_iterator()

    generic_iter = tf.data.Iterator.from_string_handle(handle, output_t, output_s)
    specific_handle = sess.run(iterator.string_handle())

    return specific_handle, iterator, generic_iter

