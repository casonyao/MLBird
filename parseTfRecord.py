import tensorflow as tf 
import numpy as np 
def _parse_record(example_proto):
    features = {
        'name': tf.FixedLenFeature((),tf.string),
        'shape': tf.FixedLenFeature([3],tf.int64),
        'data': tf.FixedLenFeature((),tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto,features=features)
    return parsed_features

def read_test(input_file):
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(_parse_record)
    iterator = dataset.make_one_shot_iterator()
   
    with tf.Session() as sess:
        features = sess.run(iterator.get_next())
        name = features['name']
        name = name.decode()
        img_data = features['data']
        shape = features['shape']

        print('-------------------')
        print(type(shape))
        print(len(img_data))

        img_data = np.fromstring(img_data,dtype=np.uint8)
        image_data = np.reshape(img_data,shape)
#        plt.figure()
#        plt.imshow(image_data)
#        plt.show()
        


        img = tf.image.encode_jpeg(image_data)
        tf.gfile.GFile('cat_encode.jpg','wb').write(img.eval())
read_test('cattfrecord')
