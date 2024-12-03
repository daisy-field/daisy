import tensorflow as tf
from keras import Model
from keras.src import regularizers
from keras.src.layers import Activation, Input, Dense, BatchNormalization, Dropout, Lambda
from keras.src.optimizers import SGD
from keras.src.utils import plot_model


#----------------------------------
#----------------tests



#----------------------------------

def hard_coded_unet_with_vae_core(self):
    inputs = Input(shape=(self.input_dim,),
                           name="encoder_input")

    # Encoder
    x1_bps = Dense(200, name="Dense_1")(inputs)
    x1 = BatchNormalization(name="Batch_1")(x1_bps)
    x1 = Activation('relu', name="Act_1")(x1)

    x2_bps = Dense(100, name="Dense_2")(x1)
    x2 = BatchNormalization(name="Batch_2")(x2_bps)
    x2 = Activation('relu', name="Act_2")(x2)

    x3_bps = Dense(10, name="Dense_3")(x2)
    x3 = BatchNormalization(name="Batch_3")(x3_bps)
    encoder_output = Activation('relu', name="Act_3")(x3)

    z_mean = Dense(self.latent_dim, name='z_mean')(encoder_output)
    z_log_var = Dense(self.latent_dim, name='z_log_var')(encoder_output)

    z = Lambda(self.sampling, name="encoder_output_sampled")([z_mean, z_log_var])

    # Decoder
    x4_bpe = Dense(10, name="Dense_4")(z)

    # Connect encoder and decoder layers
    x = tf.keras.layers.Concatenate(name="Concat_D3_D4")([x4_bpe, x3_bps])
    x4_bpe = Dense(10, name="Dens_D3_D4")(x)
    x4_bpe = BatchNormalization(name="Batch_D3_D4")(x4_bpe)
    x4 = Activation('relu', name="Act_D3_D4")(x4_bpe)

    x5_bpe = Dense(100, name="Dense_5")(x4)

    # Connect encoder and decoder layers
    x = tf.keras.layers.Concatenate(name="Concat_D2_D5")([x5_bpe, x2_bps])
    x5_bpe = Dense(100, name="Dens_D2_D5")(x)
    x5_bpe = BatchNormalization(name="Batch_D2_D5")(x5_bpe)
    x5 = Activation('relu', name="Act_D2_D5")(x5_bpe)

    x6_bpe = Dense(200, name="Dense_6")(x5)

    x = tf.keras.layers.Concatenate(name="Concat_D1_D6")([x6_bpe, x1_bps])
    x6_bpe = Dense(200, name="Dens_D1_D6")(x)
    x6_bpe = BatchNormalization(name="Batch_D1_D6")(x6_bpe)
    x6 = Activation('relu', name="Act_D1_D6")(x6_bpe)

    # Output layer
    outputs = Dense(self.input_dim, activation='sigmoid', name="decoder_output")(x6)

    # Build the U-Net model
    u_net = Model(inputs, outputs, name='u_net')

    plot_model(u_net, to_file='unet_with_vae_core_without_loss.png', show_shapes=True, show_layer_names=True)

    # Define the VAE loss function
    reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, outputs)
    reconstruction_loss *= self.input_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    u_net.add_loss(vae_loss)

    optimizer_SGD = SGD(learning_rate=self.learning_rate, momentum=0.9, clipvalue=1.0)
    u_net.compile(optimizer=optimizer_SGD)

    # Display model summary
    u_net.summary()
    plot_model(u_net, to_file='unet_with_vae_core.png', show_shapes=True, show_layer_names=True)

#----------------------------------

def looped_unet_with_vae_core(self):
    encoder_units = [200, 100, 10]
    decoder_units = [10, 100, 200]

    inputs = Input(shape=(self.input_dim,), name="encoder_input")
    x = inputs

    # Encoder
    encoder_blocks = []
    for ebc, units in enumerate(encoder_units):
        ebc += 1
        x = Dense(units, name=f"Dense_{ebc}")(x)
        encoder_blocks.append(x)
        x = BatchNormalization(name=f"Batch_{ebc}")(x)
        x = Activation('relu', name=f"Act_{ebc}")(x)

    # VAE Core
    z_mean = Dense(self.latent_dim, name='z_mean')(x)
    z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
    z = Lambda(self.sampling, name="encoder_output_sampled")([z_mean, z_log_var])

    # Decoder
    x = z
    for dbc, units in enumerate(decoder_units[0:]):
        dbc_real = dbc + 4
        dbc_reverse = -(dbc+1)
        ebc_real = dbc_real + dbc_reverse - dbc
        name_counter_part = f"D{ebc_real}_D{dbc_real}"
        x_eb = encoder_blocks[dbc_reverse]

        x = Dense(units, name=f"Dense_{dbc_real}")(x)
        x_c = tf.keras.layers.Concatenate(name="Concat_" + name_counter_part)([x, x_eb])
        x_c = Dense(units, name="Dense_" + name_counter_part)(x_c)
        x_c = BatchNormalization(name="Batch_" + name_counter_part)(x_c)
        x = Activation('relu', name="Act_" + name_counter_part)(x_c)

    # Output layer
    outputs = Dense(self.input_dim, activation='sigmoid', name="decoder_output")(x)

    # Build the U-Net model
    u_net = Model(inputs, outputs, name='u_net')

    plot_model(u_net, to_file='loop_unet_with_vae_core_without_loss.png', show_shapes=True, show_layer_names=True)

    # Define the VAE loss function
    reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, outputs)
    reconstruction_loss *= self.input_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    u_net.add_loss(vae_loss)

    optimizer_SGD = SGD(learning_rate=self.learning_rate, momentum=0.9, clipvalue=1.0)
    u_net.compile(optimizer=optimizer_SGD)

    # Display model summary
    u_net.summary()
    plot_model(u_net, to_file='loop_unet_with_vae_core.png', show_shapes=True, show_layer_names=True)

#----------------------------------

def looped_unet_with_vae_core_original(self):
    encoder_units = [200, 100, 10]
    decoder_units = [10, 100, 200]

    inputs = Input(shape=(self.input_dim,), name="encoder_input")

    #hier
    z_mean, z_log_var, encoder_blocks = self.build_encoder_2(inputs, encoder_units)

    z = Lambda(self.sampling, name="encoder_output_sampled")([z_mean, z_log_var])

    outputs = self.build_decoder_2(z,  decoder_units, encoder_blocks)

    # Build the encoder model
    encoder_model = Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='encoder_model')
    encoder_model.summary()
    plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True, show_layer_names=True)

    # Build the decode model
    latent_inputs = Input(shape=(self.latent_dim,), name="decoder_input")
    z_mean_2, z_log_var_2, encoder_blocks_2 = self.build_encoder_2(latent_inputs, encoder_units)
    outputs_2 = self.build_decoder_2(latent_inputs, decoder_units, encoder_blocks_2)

    decoder_model = Model(inputs=latent_inputs, outputs=outputs_2, name='decoder_model')
    decoder_model.summary()
    plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True, show_layer_names=True)

    # Build the U-Net model
    u_net = Model(inputs=inputs, outputs=outputs, name='u_net')
    u_net.summary()
    plot_model(u_net, to_file='loop_unet_with_vae_core_without_loss.png', show_shapes=True, show_layer_names=True)

    # Define the VAE loss function
    reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, outputs)
    reconstruction_loss *= self.input_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    u_net.add_loss(vae_loss)

    optimizer_SGD = SGD(learning_rate=self.learning_rate, momentum=0.9, clipvalue=1.0)
    u_net.compile(optimizer=optimizer_SGD)

    # Display model summary
    u_net.summary()
    plot_model(u_net, to_file='loop_unet_with_vae_core.png', show_shapes=True, show_layer_names=True)

def build_encoder_2(self, inputs, encoder_units):
    x = inputs

    reg_losses_encoder = []

    # Encoder
    encoder_blocks = []
    for ebc, units in enumerate(encoder_units):
        ebc += 1
        dense_layer = Dense(units, name=f"Dense_{ebc}")
        x = dense_layer(x)

        encoder_blocks.append(x)

        if self.use_batch_norm:
            x = BatchNormalization(name=f"Batch_{ebc}")(x)

        x = Activation('relu', name=f"Act_{ebc}")(x)

        if self.use_dropout:
            x = Dropout(0.2, name=f"decoder_dropout_unit_{ebc}")(x)

        # Add regularization loss if trainable weights exist
        if dense_layer.trainable_weights:
            reg_losses_encoder.append(
                regularizers.l2(0.01)(dense_layer.kernel)
            )

    # VAE Core
    z_mean = Dense(self.latent_dim, name='z_mean')(x)
    z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

    return z_mean, z_log_var, encoder_blocks

def build_decoder_2(self, inputs,  decoder_units, encoder_blocks):
    reg_losses_decoder = []

    # Decoder
    x = inputs

    for dbc, units in enumerate(decoder_units[0:]):
        dbc_real = dbc + 4
        dbc_reverse = -(dbc + 1)
        ebc_real = dbc_real + dbc_reverse - dbc

        name_counter_part = f"D{ebc_real}_D{dbc_real}"
        x_eb = encoder_blocks[dbc_reverse]

        dense_layer = Dense(units, name=f"Dense_{dbc_real}")
        x = dense_layer(x)
        x_c = tf.keras.layers.Concatenate(name=f"Concat_{name_counter_part}")([x, x_eb])
        x_c = Dense(units, name=f"Dense_{name_counter_part}")(x_c)

        if self.use_batch_norm:
            x_c = BatchNormalization(name=f"Batch_{name_counter_part}")(x_c)

        x_c = Activation('relu', name=f"Act_{name_counter_part}")(x_c)

        if self.use_dropout:
            x_c = Dropout(0.2, name=f"decoder_dropout_unit_{name_counter_part}")(x_c)

        x = x_c

        if dense_layer.trainable_weights:
            reg_losses_decoder.append(
                regularizers.l2(0.01)(dense_layer.kernel)
            )

    # Output layer
    outputs = Dense(self.input_dim, activation='sigmoid', name="decoder_output")(x)

    return outputs

#----------------------------------

def looped_unet_with_vae_core_original(self):
    encoder_units = [200, 100, 10]
    decoder_units = [10, 100, 200]

    inputs = Input(shape=(self.input_dim,), name="encoder_input")

    #hier
    z_mean, z_log_var, encoder_blocks = self.build_encoder_2(inputs, encoder_units)

    encoder_model = Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder_model')
    encoder_model.summary()
    plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True, show_layer_names=True)

    encoder_out = encoder_model(inputs)
    z_mean, z_log_var = tf.split(encoder_out, num_or_size_splits=2, axis=-1)

    z = Lambda(self.sampling, name="encoder_output_sampled")([z_mean, z_log_var])

    # Build the decode model
    latent_inputs = Input(shape=(self.latent_dim,), name="decoder_input")
    z_mean_2, z_log_var_2, encoder_blocks_2 = self.build_encoder_2(latent_inputs, encoder_units)
    outputs_2 = self.build_decoder_2(latent_inputs, decoder_units, encoder_blocks_2)

    decoder_model = Model(inputs=latent_inputs, outputs=outputs_2, name='decoder_model')
    decoder_model.summary()
    plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True, show_layer_names=True)

    reconstructed = decoder_model(z)

    # Build the U-Net model
    u_net = Model(inputs=inputs, outputs=reconstructed, name='u_net')
    u_net.summary()
    plot_model(u_net, to_file='loop_unet_with_vae_core_without_loss.png', show_shapes=True, show_layer_names=True)

    encoder_model = u_net.get_layer('encoder_model')

    # Define the VAE loss function
    reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, reconstructed)
    reconstruction_loss *= self.input_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    u_net.add_loss(vae_loss)

    optimizer_SGD = SGD(learning_rate=self.learning_rate, momentum=0.9, clipvalue=1.0)
    u_net.compile(optimizer=optimizer_SGD)

    # Display model summary
    u_net.summary()
    plot_model(u_net, to_file='loop_unet_with_vae_core.png', show_shapes=True, show_layer_names=True)

def build_encoder_2(self, inputs, encoder_units):
    x = inputs

    reg_losses_encoder = []

    # Encoder
    encoder_blocks = []
    for ebc, units in enumerate(encoder_units):
        ebc += 1
        dense_layer = Dense(units, name=f"Dense_{ebc}")
        x = dense_layer(x)

        encoder_blocks.append(x)

        if self.use_batch_norm:
            x = BatchNormalization(name=f"Batch_{ebc}")(x)

        x = Activation('relu', name=f"Act_{ebc}")(x)

        if self.use_dropout:
            x = Dropout(0.2, name=f"decoder_dropout_unit_{ebc}")(x)

        # Add regularization loss if trainable weights exist
        if dense_layer.trainable_weights:
            reg_losses_encoder.append(
                regularizers.l2(0.01)(dense_layer.kernel)
            )

    # VAE Core
    z_mean = Dense(self.latent_dim, name='z_mean')(x)
    z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

    return z_mean, z_log_var, encoder_blocks

def build_decoder_2(self, inputs,  decoder_units, encoder_blocks):
    reg_losses_decoder = []

    # Decoder
    x = inputs

    for dbc, units in enumerate(decoder_units[0:]):
        dbc_real = dbc + 4
        dbc_reverse = -(dbc + 1)
        ebc_real = dbc_real + dbc_reverse - dbc

        name_counter_part = f"D{ebc_real}_D{dbc_real}"
        x_eb = encoder_blocks[dbc_reverse]

        dense_layer = Dense(units, name=f"Dense_{dbc_real}")
        x = dense_layer(x)
        x_c = tf.keras.layers.Concatenate(name=f"Concat_{name_counter_part}")([x, x_eb])
        x_c = Dense(units, name=f"Dense_{name_counter_part}")(x_c)

        if self.use_batch_norm:
            x_c = BatchNormalization(name=f"Batch_{name_counter_part}")(x_c)

        x_c = Activation('relu', name=f"Act_{name_counter_part}")(x_c)

        if self.use_dropout:
            x_c = Dropout(0.2, name=f"decoder_dropout_unit_{name_counter_part}")(x_c)

        x = x_c

        if dense_layer.trainable_weights:
            reg_losses_decoder.append(
                regularizers.l2(0.01)(dense_layer.kernel)
            )

    # Output layer
    outputs = Dense(self.input_dim, activation='sigmoid', name="decoder_output")(x)

    return outputs

#-----------------------------------

def looped_unet_with_vae_core_original(self):
    encoder_units = [200, 100, 10]
    decoder_units = [10, 100, 200]

    inputs = Input(shape=(self.input_dim,), name="encoder_input")

    #hier
    z_mean, z_log_var, encoder_blocks = self.build_encoder_2(inputs, encoder_units)

    encoder_model = Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder_model')
    encoder_model.summary()
    plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True, show_layer_names=True)

    encoder_out = encoder_model(inputs)
    z_mean, z_log_var = tf.split(encoder_out, num_or_size_splits=2, axis=-1)

    z = Lambda(self.sampling, name="encoder_output_sampled")([z_mean, z_log_var])

    # Build the decode model
    # latent_inputs = Input(shape=(self.latent_dim,), name="decoder_input")
    z_mean_2, z_log_var_2, encoder_blocks_2 = self.build_encoder_2(z, encoder_units)
    outputs_2 = self.build_decoder_2(z, decoder_units, encoder_blocks_2)

    decoder_model = Model(inputs=z, outputs=outputs_2, name='decoder_model')
    decoder_model.summary()
    plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True, show_layer_names=True)

    reconstructed = decoder_model(z)

    # Build the U-Net model
    u_net = Model(inputs=inputs, outputs=reconstructed, name='u_net')
    u_net.summary()
    plot_model(u_net, to_file='loop_unet_with_vae_core_without_loss.png', show_shapes=True, show_layer_names=True)

    encoder_model = u_net.get_layer('encoder_model')

    # Define the VAE loss function
    reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, reconstructed)
    reconstruction_loss *= self.input_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    u_net.add_loss(vae_loss)

    optimizer_SGD = SGD(learning_rate=self.learning_rate, momentum=0.9, clipvalue=1.0)
    u_net.compile(optimizer=optimizer_SGD)

    # Display model summary
    u_net.summary()
    plot_model(u_net, to_file='loop_unet_with_vae_core.png', show_shapes=True, show_layer_names=True)

def build_encoder_2(self, inputs, encoder_units):
    x = inputs

    reg_losses_encoder = []

    # Encoder
    encoder_blocks = []
    for ebc, units in enumerate(encoder_units):
        ebc += 1
        dense_layer = Dense(units, name=f"Dense_{ebc}")
        x = dense_layer(x)

        encoder_blocks.append(x)

        if self.use_batch_norm:
            x = BatchNormalization(name=f"Batch_{ebc}")(x)

        x = Activation('relu', name=f"Act_{ebc}")(x)

        if self.use_dropout:
            x = Dropout(0.2, name=f"decoder_dropout_unit_{ebc}")(x)

        # Add regularization loss if trainable weights exist
        if dense_layer.trainable_weights:
            reg_losses_encoder.append(
                regularizers.l2(0.01)(dense_layer.kernel)
            )

    # VAE Core
    z_mean = Dense(self.latent_dim, name='z_mean')(x)
    z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

    return z_mean, z_log_var, encoder_blocks

def build_decoder_2(self, inputs,  decoder_units, encoder_blocks):
    reg_losses_decoder = []

    # Decoder
    x = inputs

    for dbc, units in enumerate(decoder_units[0:]):
        dbc_real = dbc + 4
        dbc_reverse = -(dbc + 1)
        ebc_real = dbc_real + dbc_reverse - dbc

        name_counter_part = f"D{ebc_real}_D{dbc_real}"
        x_eb = encoder_blocks[dbc_reverse]

        dense_layer = Dense(units, name=f"Dense_{dbc_real}")
        x = dense_layer(x)
        x_c = tf.keras.layers.Concatenate(name=f"Concat_{name_counter_part}")([x, x_eb])
        x_c = Dense(units, name=f"Dense_{name_counter_part}")(x_c)

        if self.use_batch_norm:
            x_c = BatchNormalization(name=f"Batch_{name_counter_part}")(x_c)

        x_c = Activation('relu', name=f"Act_{name_counter_part}")(x_c)

        if self.use_dropout:
            x_c = Dropout(0.2, name=f"decoder_dropout_unit_{name_counter_part}")(x_c)

        x = x_c

        if dense_layer.trainable_weights:
            reg_losses_decoder.append(
                regularizers.l2(0.01)(dense_layer.kernel)
            )

    # Output layer
    outputs = Dense(self.input_dim, activation='sigmoid', name="decoder_output")(x)

    return outputs
