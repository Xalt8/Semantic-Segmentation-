import tensorflow as tf

def conv_block(input:tf.Tensor, num_filters:int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def encoder_block(input:tf.Tensor, num_filters:int) -> tf.Tensor:
    x = conv_block(input=input, num_filters=num_filters)
    p = tf.keras.layers.MaxPool2D((2,2))(x)
    return x, p


def decoder_block(input:tf.Tensor, skip_features:tf.Tensor, num_filters:int) -> tf.Tensor:
    x = tf.keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=(2,2), strides=2, padding='same')(input)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def multi_class_unet(n_classes:int, input_shape:tuple) -> tf.keras.Model:
    inputs = tf.keras.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(input=inputs, num_filters=64)
    s2, p2 = encoder_block(input=p1, num_filters=128)
    s3, p3 = encoder_block(input=p2, num_filters=256)
    s4, p4 = encoder_block(input=p3, num_filters=512)
    # Bridge
    b1 = conv_block(input=p4, num_filters=1024)
    # Decoder
    d1 = decoder_block(input=b1, skip_features=s4, num_filters=512)
    d2 = decoder_block(input=d1, skip_features=s3, num_filters=256)
    d3 = decoder_block(input=d2, skip_features=s2, num_filters=128)
    d4 = decoder_block(input=d3, skip_features=s1, num_filters=64)

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='softmax')(d4)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == '__main__':
    
    model = multi_class_unet(n_classes=6, input_shape=(128,128,3))
    print(model.summary())

