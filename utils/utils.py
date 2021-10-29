import copy
import tensorflow as tf

def summary(model, input_shape, image=True):
    dummy_model = copy.deepcopy(model)
    tmp_x = tf.keras.Input(shape=input_shape, name='summary_input')
    tmp_m = tf.keras.Model(inputs=tmp_x, outputs=dummy_model.call(tmp_x), name='summary_model')
    return tf.keras.utils.plot_model(tmp_m, show_shapes=True, dpi=64) if image else tmp_m.summary()