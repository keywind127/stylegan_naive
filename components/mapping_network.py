from tensorflow.keras.layers import LeakyReLU, Dense, Input # type:ignore
from tensorflow.keras.models import Sequential, Model # type:ignore 

def create_mapping_network(num_layers : int, vector_size : int) -> Model:

    assert isinstance(num_layers, int)
    assert isinstance(vector_size, int)

    return Sequential([
        Input(shape = (vector_size,))
    ] + [
        layer for _ in range(num_layers)
            for layer in (Dense(vector_size), LeakyReLU(0.02))
    ])

if (__name__ == "__main__"):

    mapping_network = create_mapping_network(8, 512)

    mapping_network.summary()