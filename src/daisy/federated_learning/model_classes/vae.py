import tensorflow as tf
from tensorflow import keras
from keras.layers import concatenate

class DetectorVAE:
    def __init__(self, input_dim, hidden_layers, latent_dim, unet_switch = False):
        self.is_unet = unet_switch

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim

        self.encoder_layers = []

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model, self.loss = self.build_vae()

    def build_encoder(self):
        # Encoder
        inputs = keras.layers.Input(shape=(self.input_dim,))
        h = inputs

        # Dynamisch versteckte Schichten im Encoder hinzufügen
        for idx, units in enumerate(self.hidden_layers):
            if idx == 0:
                h = keras.layers.Dense(units, activation="relu", activity_regularizer=keras.regularizers.l1(10e-5))(h)
            else:
                h = keras.layers.Dense(units, activation="relu")(h)

            h = keras.layers.Dropout(0.1)(h)

            self.encoder_layers.append(h)

        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(h)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(h)
        z = Sampling()([z_mean, z_log_var])

        return keras.models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self):
        # Decoder
        latent_inputs = keras.layers.Input(shape=(self.latent_dim,))
        h = latent_inputs

        # Dynamisch versteckte Schichten im Decoder hinzufügen
        for idx, units in enumerate(reversed(self.hidden_layers)):
            h = keras.layers.Dense(units, activation='relu')(h)
            h = keras.layers.Dropout(0.1)(h)

            # Skip-Connection => only for unet
            if self.is_unet:
                h = concatenate([h, self.encoder_layers[-(idx + 1)]])

        h = keras.layers.Dense(self.input_dim, activation='tanh')(h)

        return keras.models.Model(latent_inputs, h, name="decoder")

    def build_vae(self):
        inputs = keras.layers.Input(shape=self.input_dim)
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)

        model = keras.models.Model(inputs, outputs, name="vae")
        loss = DetectorVAE.vae_loss(inputs, outputs, z_log_var, z_mean)
        return model, loss

    @staticmethod
    # Eigenständige Schicht für den Verlust
    def vae_loss(x, x_decoded_mean, z_log_var, z_mean):
        xent_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
        #xent_loss *= input_dim
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return tf.reduce_mean(xent_loss + kl_loss)


    # Custom Sampling Layer
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]  # Use tf.shape instead of K.shape
        dim = tf.shape(z_mean)[1]    # Use tf.shape instead of K.shape
        epsilon = tf.random.normal(shape=(batch, dim))  # Replace K.random_normal with tf.random.normal
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#class VaeLoss(keras.layers.Layer):
#    def __init__(self, input_dim, **kwargs):
#        super(VaeLoss, self).__init__(**kwargs)
#        self.input_dim = input_dim
#
#    # Custom loss layer
#    def call(self, true, pred):
#        pred_mean, z_mean, z_log_var = pred
#
#        # Reconstruction loss
#        xent_loss = mean_squared_error(true, pred_mean)
#        xent_loss *= self.input_dim
#
#        # KL divergence loss
#        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)  # Use tf.square and tf.exp
#        kl_loss = tf.reduce_sum(kl_loss, axis=-1)  # Use tf.reduce_sum instead of K.sum
#        kl_loss *= -0.5
#
#        # Combine both losses
#        return tf.reduce_mean(xent_loss + kl_loss)  # Use tf.reduce_mean instead of K.mean


