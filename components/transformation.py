from tensorflow.keras.layers import Reshape, Input, Dense # type:ignore
from tensorflow.keras.models import Sequential, Model # type:ignore
from typing import Tuple
import tensorflow as tf

def create_affine_transform(input_shape  : int, 
                            output_shape : Tuple[ int, int ]) -> Model:
    
    assert isinstance(input_shape, int)
    assert isinstance(output_shape, tuple)
    
    return Sequential([
        Input(shape = (input_shape, )),
        Dense(tf.math.reduce_prod(output_shape), activation = None),
        Reshape(output_shape)
    ])

if (__name__ == "__main__"):

    affine_transform = create_affine_transform(512, (512, 2))

    affine_transform.summary()