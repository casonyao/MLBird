import tensorflow as tf 
def write_test(input,output):
    writer = tf.python_io.TFRecordWriter(output)
    image = tf.read_file(input)
    image = tf.image.decode_jpeg(image)
    
    with tf.Session() as sess:
        image = sess.run(image)
        shape = image.shape
        image_data = image.tostring()
        print(type(image))
        print(len(image_data))
        name = bytes("cat",encoding='utf8')
        print(type(name))
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0],shape[1],shape[2]])),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
       
        }
        ))
        writer.write(example.SerializeToString())
        writer.close()

write_test('cat.jpg','cattfrecord')
