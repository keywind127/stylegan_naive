from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Softmax, LeakyReLU

def create_base_discriminator() -> Model:

    input_layer = Input(shape = (4, 4, 3))

    outputs = Conv2D(32, (3, 3), activation = LeakyReLU(0.02))(input_layer)

    outputs = Flatten()(outputs)

    outputs = Dense(128, activation = LeakyReLU(0.02))(outputs)

    outputs = Dense(2, activation = LeakyReLU(0.02))(outputs)

    outputs = Softmax()(outputs)

    return Model(inputs = [ input_layer ], outputs = [ outputs ])

if (__name__ == "__main__"):

    discriminator = create_base_discriminator()

    discriminator.summary()