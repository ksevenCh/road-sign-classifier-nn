import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('models/model2')

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tfl_model = converter.convert()

with open('model1.tflite', 'wb') as f:
    f.write(tfl_model)