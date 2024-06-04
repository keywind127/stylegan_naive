from tensorflow.keras.layers import Layer, Input # type:ignore
from tensorflow.keras.models import Model # type:ignore
from tensorflow.keras import backend as K # type:ignore
from typing import Iterable, Tuple
import tensorflow as tf

class AdaIN(Layer):

    def __init__(self, *args, **kwargs) -> None:
        super(AdaIN, self).__init__(*args, **kwargs)

    def call(self, inputs : Iterable[ tf.Tensor ]) -> tf.Tensor:

        # (B, C, 2)
        style_tensor = inputs[1]

        # (B, H, W, C)
        input_tensor = inputs[0]

        # (B, H, W, C)
        mean_input, var_input = tf.nn.moments(input_tensor, axes = [ 1, 2 ], keepdims = True)

        # (B, H, W, C)
        sigma_input = tf.math.sqrt(var_input + K.epsilon())

        # (B, H, W, C)
        input_tensor = (input_tensor - mean_input) / sigma_input

        # (B, H, W, C)
        input_tensor = input_tensor * tf.reshape(style_tensor[..., 0:1], (-1, 1, 1, 1)) + tf.reshape(style_tensor[..., 1:2], (-1, 1, 1, 1))

        return input_tensor
    
def create_adain(input_shape : Tuple[ int, int, int ], 
                 style_shape : Tuple[ int, int ]
            ) -> Model:
    
    assert isinstance(input_shape, tuple)
    assert isinstance(style_shape, tuple)
    
    content_input = Input(shape = input_shape, name = 'content_input')
    
    style_input = Input(shape = style_shape, name = 'style_input')

    adain_output = AdaIN(name = "AdaIN")([content_input, style_input])

    model = Model(inputs = [content_input, style_input], outputs = adain_output)

    return model
    
if (__name__ == "__main__"):

    adain_network = create_adain((4, 4, 512), (512, 2))

    adain_network.summary()