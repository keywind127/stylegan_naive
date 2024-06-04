# from tensorflow.keras.layers import Layer 
# from typing import Optional, Tuple
# import tensorflow as tf

# class ConstantLatent(Layer):
    
#     def __init__(self, output_shape : Tuple[ int, int, int ], 
#                        use_random   : Optional[ bool ] = False, **kwargs) -> None:
        
#         assert isinstance(output_shape, tuple)
#         assert isinstance(use_random, bool)

#         super (ConstantLatent, self).__init__(**kwargs)
#         self.start_tensor = tf.random.normal(output_shape) if (use_random) else tf.ones(output_shape)

#     def call(self) -> tf.Tensor:
#         return self.start_tensor
    
# def create_constant(output_shape : Tuple[ int, int, int ], use_random : Optional[ bool ] = False) -> Layer:

#     return ConstantLatent(output_shape, use_random)

# if (__name__ == "__main__"):

#     from tensorflow.keras.models import Model

#     from tensorflow.keras.layers import Lambda, Input

#     from tensorflow.keras import backend as K

#     input_tensor = Input(shape = (4, 4, 512))

#     constant_tensor = Input(tensor = K.variable(tf.random.normal((4, 4, 512))))
    
#     @tf.function
#     def add_constant(x) -> tf.Tensor:
#         return tf.reshape(x[1], tf.shape(x[1])) + x[0]

#     output_tensor = Lambda(add_constant)([ input_tensor, constant_tensor ])

#     model = Model(inputs = [ input_tensor, constant_tensor ], outputs = [ output_tensor ])

#     model.summary()

#     model([ tf.random.uniform((32, 4, 4, 512)) ])