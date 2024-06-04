from tensorflow.keras.layers import UpSampling2D, Conv2D, Input # type:ignore
from tensorflow.keras.models import Model # type:ignore

from components.weighted_channels import create_weighted_channels_model
from components.transformation import create_affine_transform
from components.normalization import create_adain

from typing import Optional

def create_base_generator(height   : Optional[ int ] = 4, 
                          width    : Optional[ int ] = 4, 
                          channels : Optional[ int ] = 512) -> Model:
    
    assert isinstance(height, int)
    assert isinstance(width, int)
    assert isinstance(channels, int)
        
    WC1 = create_weighted_channels_model(height, width, channels)

    WC2 = create_weighted_channels_model(height, width, channels)

    ADAIN = create_adain((height, width, channels), (channels, 2))

    AT1 = create_affine_transform(channels, (channels, 2))
    
    AT2 = create_affine_transform(channels, (channels, 2))

    CONV1 = Conv2D(channels, (3, 3), padding = "same", activation = None)

    const_layer = Input(shape = (height, width, channels))

    noise_layers = [
        Input(shape = (height, width, channels)),
        Input(shape = (height, width, channels))
    ]

    style_layers = [
        Input(shape = (channels,)),
        Input(shape = (channels,))
    ]

    outputs = WC1([ const_layer, noise_layers[0] ])

    outputs = ADAIN([ outputs, AT1(style_layers[0]) ])

    outputs = CONV1(outputs)

    outputs = WC2([ outputs, noise_layers[1] ])

    outputs = ADAIN([ outputs, AT2(style_layers[1]) ])

    model = Model(inputs = [ const_layer ] + noise_layers + style_layers, outputs = [ outputs ])

    return model

if (__name__ == "__main__"):

    base_generator = create_base_generator(4, 4, 512)

    base_generator.summary()