from tensorflow.keras.layers import Layer, Input # type:ignore
from tensorflow.keras.models import Model # type:ignore
from typing import Iterable
import tensorflow as tf

class WeightedChannels(Layer):

    def __init__(self, num_channels : int, *args, **kwargs) -> None:

        assert isinstance(num_channels, int)

        super(WeightedChannels, self).__init__(*args, **kwargs)
        self.num_channels = num_channels
        self._weights = self.add_weight(
            name        = "channel_weights",
            shape       = (num_channels,),
            initializer = "uniform",
            trainable   = True
        )

    def call(self, inputs : Iterable[ tf.Tensor ]) -> tf.Tensor:
        
        # (B, H, W, C)
        input_tensor = inputs[0]

        # (B, H, W, C)
        noise_tensor = inputs[1]

        # (B, H, W, C)
        input_tensor = input_tensor + noise_tensor * self._weights

        return input_tensor
    
def create_weighted_channels_model(height : int, width : int, channels : int) -> Model:

    assert isinstance(height, int)
    assert isinstance(width, int)
    assert isinstance(channels, int)

    input_layer = Input(shape = (height, width, channels))
    noise_layer = Input(shape = (height, width, channels))

    output_layer = WeightedChannels(channels)([ input_layer, noise_layer ])

    model = Model(inputs = [ input_layer, noise_layer ], outputs = [ output_layer ])

    return model
    
if (__name__ == "__main__"):

    model = create_weighted_channels_model(4, 4, 512)

    model.summary()

    noise_tensor = tf.random.uniform((32, 4, 4, 512))

    input_tensor = tf.random.uniform((32, 4, 4, 512))

    model([ input_tensor, noise_tensor ])