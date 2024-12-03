import tensorflow as tf
from keras import Model
from keras.src.layers import Activation, Input, Dense, BatchNormalization
from keras.src.utils import plot_model
import pandas as pd
import os
import ast
import re
import numpy as np
class HardCodedUnetAutoencoder():
    def __init__(self):
        tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
        print("Visible devices:" + str(tf.config.get_visible_devices()) + " all gpus: " + str(tf.config.list_physical_devices('GPU')))

        self.autoencoder = None
        self.input_dim = 0
        self.train_test_data = None

    def test_hard_coded_unet(self, train_test_data):
        self.train_test_data = train_test_data
        self.train_data = self.train_test_data.drop(columns=['meta.number', 'meta.time_epoch', 'meta.time', 'DATA.data', 'DATA.data.data', 'tls.handshake.random_time', 'label','eth._ws.expert.message', 'tls.handshake.ja3_full', 'tls.handshake.ja3', 'tls.handshake.ja3s_full', 'tls.handshake.ja3s', 'tcp.flags.str'])
        self.train_data = self.preprocess_train_data(self.train_data)
        self.test_data = self.train_test_data[['label']]
        self.input_dim = self.train_data.shape[1]

        self.autoencoder = self.construct_hard_coded_unet()

    def is_list_format(self, s):
        return re.match(r"^\[.*\]$", str(s)) is not None

    def is_colon_separated_format(self, s):
        return re.match(r"^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2})+$", str(s)) is not None

    def is_abbreviated_ipv6_format(self, s):
        # Dieser reguläre Ausdruck erkennt nur abgekürzte IPv6-Adressen, die "::" enthalten
        return re.match(r"^((:[0-9a-fA-F]{1,4}){0,7}|::[0-9a-fA-F]{0,4})$", str(s)) is not None

    def is_abbreviated_ipv6_list_format(self, s):
        # Check if the string is a valid list of IPv6 addresses
        try:
            addresses = eval(s)
            if not isinstance(addresses, list):
                return False
            return all(self.is_abbreviated_ipv6_format(addr) for addr in addresses)
        except:
            return False

    def is_ipv4_format(self, s):
        # Dieser reguläre Ausdruck erkennt nur abgekürzte IPv6-Adressen, die "::" enthalten
        ipv4_regex = re.compile(r"^(25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])(\.(25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])){3}$")
        return ipv4_regex.match(str(s)) is not None

    def is_TCP_Flags(self, s):
        # Dieser reguläre Ausdruck erkennt nur abgekürzte IPv6-Adressen, die "::" enthalten
        if len(s) == 12:
            tcp_flags_regex = re.compile(r"^(?:[·UAPRSFENC]{0,2}){9}$")
            return tcp_flags_regex.match(str(s)) is not None
        else:
            return False

    # Helper function to split colon-separated strings into separate columns
    def split_colon_separated_values(self, df, col_name, new_col_prefix):
        split_values = df[col_name].str.split(':', expand=True)
        for i, col in enumerate(split_values.columns):
            df[f"{new_col_prefix}.{i}"] = split_values[col]

    def split_ipv6_into_hextets(self, df, ipv6_col_name):
        # Funktion zum Aufteilen der IPv6-Adresse in Hextets
        def split_ipv6(address):
            hextets = address.split(':')
            hextets_padded = [hextet.zfill(4) if hextet != '' else '0000' for hextet in hextets]
            return hextets_padded

        # Vorbereiten zum ersetze NaN-Werte in den Zeilen mit dem Wert 0 durch "0000"
        df.loc[df[ipv6_col_name] == '0', ipv6_col_name] = '::'
        # Neuen DataFrame erstellen, der die aufgeteilten Hextets enthält
        hextet_df = df[ipv6_col_name].apply(split_ipv6).apply(pd.Series)
        # Spaltennamen für die Hextets erstellen
        hextet_columns = [f'{ipv6_col_name}_hextet_{i}' for i in range(1, len(hextet_df.columns) + 1)]
        # Spaltennamen festlegen und die ursprüngliche IPv6-Spalte entfernen
        hextet_df.columns = hextet_columns
        df = pd.concat([df, hextet_df], axis=1)
        return df

    def convert_to_list(self, val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    def split_ipv6_list_into_hextets(self, df, ipv6_list_col_name):
        # Funktion zum Aufteilen der IPv6-Adresse in Hextets
        def split_ipv6(address):
            hextets = address.split(':')
            hextets_padded = [hextet.zfill(4) if hextet != '' else '0000' for hextet in hextets]
            return hextets_padded

        # Vorbereiten zum ersetze NaN-Werte in den Zeilen mit dem Wert 0 durch "0000"
        df.loc[df[ipv6_list_col_name] == '0', ipv6_list_col_name] = '[\'::\',\'::\']'

        # convert list to single columns
        df[ipv6_list_col_name] = df[ipv6_list_col_name].apply(self.convert_to_list)
        df_expanded = df[ipv6_list_col_name].apply(pd.Series)
        df_expanded.columns = [f'{ipv6_list_col_name}_{i + 1}' for i in df_expanded.columns]
        #df = pd.concat([df, df_expanded], axis=1).drop(columns=[ipv6_list_col_name])


        for df_expanded_col in df_expanded.columns:
            # Neuen DataFrame erstellen, der die aufgeteilten Hextets enthält
            hextet_df = df_expanded[df_expanded_col].apply(split_ipv6).apply(pd.Series)
            # Spaltennamen für die Hextets erstellen
            hextet_columns = [f'{df_expanded_col}_hextet_{i}' for i in range(1, len(hextet_df.columns) + 1)]
            # Spaltennamen festlegen und die ursprüngliche IPv6-Spalte entfernen
            hextet_df.columns = hextet_columns
            df = pd.concat([df, hextet_df], axis=1)
        return df

    def split_ip_address(self, ip):
        octets = ip.split('.')
        return [int(octet) for octet in octets]
    def split_ipv4_into_octets(self, df, col):
        # Vorbereiten zum ersetze NaN-Werte in den Zeilen mit dem Wert 0 durch "0000"
        df.loc[df[col] == '0', col] = '0.0.0.0'

        octets_df = df[col].apply(lambda x: pd.Series(self.split_ip_address(x)))
        octets_df.columns = [f'{col}_{i + 1}' for i in range(octets_df.shape[1])]

        df = pd.concat([df, octets_df], axis=1)

        return df

    def replace_nans(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna('0', inplace=True)
            elif np.issubdtype(df[col].dtype, np.number):
                df[col].fillna(0, inplace=True)
            else:
                raise Exception("Not implemented type in replace_nans")

    # Konvertierung von Hex-Strings in Integer-Werte
    def hex_to_int(self, value):
        if isinstance(value, str) and value.startswith("0x"):
            try:
                return int(value, 16)
            except ValueError:
                return value
        else:
            return value

    # Funktion zum Konvertieren der Werte basierend auf dem Spaltentyp
    def convert_values(self, value, dtype):
        if dtype == 'object':
            return self.hex_to_int(value)
        return value
    # Konvertierung der Werte basierend auf dem Spaltentyp
    def cast_hexstr_to_int(self, df):
        for col, dtype in df.dtypes.items():
            df[col] = df[col].apply(lambda x: self.convert_values(x, dtype))
    def preprocess_train_data(self, df):
        df = df.dropna(axis=1, how='all')
        #https://datagy.io/pandas-fillna/
        self.cast_hexstr_to_int(df)
        self.replace_nans(df)

        # transfer list columns to on hot encoding
        list_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str)).all()]
        # type(df['sll.ltype'].iloc[14])
        # type(df['sll.ltype'].iloc[15])

        for col in list_columns:

            if df[col].apply(self.is_list_format).any():
                # Flache Liste aller Werte in der Spalte erstellen
                all_values = set(val for sublist in df[col].apply(ast.literal_eval) for val in sublist)

                # Für jeden Wert eine neue Spalte erstellen und mit 1 oder 0 befüllen
                for value in all_values:
                    df.loc[:, f"{col}.{value}"] = df[col].apply(lambda x: 1 if value in ast.literal_eval(x) else 0)

            elif df[col].apply(self.is_colon_separated_format).any():
                self.split_colon_separated_values(df, col, col)

            elif df[col].apply(self.is_abbreviated_ipv6_format).any():
                df = self.split_ipv6_into_hextets(df, col)

            elif df[col].apply(self.is_abbreviated_ipv6_list_format).any():
                df = self.split_ipv6_list_into_hextets(df, col)

            elif df[col].apply(self.is_ipv4_format).any():
                df = self.split_ipv4_into_octets(df, col)

            else:
                raise Exception("Not implemented coverter in preprocess_train_data")

            # Originalspalte entfernen
            df.drop(columns=[col], inplace=True)

        return df

    def construct_hard_coded_unet(self):
        inputs = Input(shape=(self.input_dim,),
                       name="encoder_input")

        # Encoder
        x = Dense(200, name="Dense_1")(inputs)
        x1 = BatchNormalization(name="Batch_1")(x)
        x1 = Activation('relu', name="Act_1")(x1)

        x = Dense(100, name="Dense_2")(x1)
        x2 = BatchNormalization(name="Batch_2")(x)
        x2 = Activation('relu', name="Act_2")(x2)

        x = Dense(10, name="Dense_3")(x2)
        x3 = BatchNormalization(name="Batch_3")(x)
        encoder_output = Activation('relu', name="Act_3")(x3)

        # Decoder
        x = Dense(10, name="Dense_4")(encoder_output)
        x4 = BatchNormalization(name="Batch_4")(x)
        x4 = Activation('relu', name="Act_4")(x4)

        x = Dense(100, name="Dense_5")(x4)
        x5 = BatchNormalization(name="Batch_5")(x)
        x5 = Activation('relu', name="Act_5")(x5)

        # Connect encoder and decoder layers
        decoder_input = tf.keras.layers.Concatenate(name="Concat_D2_D5")([x5, x2])
        x5 = Dense(100, name="Dens_D2_D5")(decoder_input)
        x5 = BatchNormalization(name="Batch_D2_D5")(x5)
        x5 = Activation('relu', name="Act_D2_D5")(x5)

        x = Dense(200, name="Dense_6")(x5)
        x6 = BatchNormalization(name="Batch_6")(x)
        x6 = Activation('relu', name="Act_6")(x6)

        decoder_input = tf.keras.layers.Concatenate(name="Concat_D1_D6")([x6, x1])
        x6 = Dense(200, name="Dens_D1_D6")(decoder_input)
        x6 = BatchNormalization(name="Batch_D1_D6")(x6)
        x6 = Activation('relu', name="Act_D1_D6")(x6)

        # Output layer
        outputs = Dense(self.input_dim, activation='sigmoid')(x6)

        # Build the U-Net model
        u_net = Model(inputs, outputs, name='u_net')

        # Display model summary
        u_net.summary()
        plot_model(u_net, to_file='u_net_plot.png', show_shapes=True, show_layer_names=True)
        return u_net

    def train_hard_coded_unet(self):
        self.autoencoder

def main():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'csv-example.csv')

    sample_df = pd.read_csv(csv_path)
    column_types = sample_df.dtypes.to_dict()
    del sample_df

    df = pd.read_csv(csv_path, dtype=column_types)

    hard_coded_unet_autoencoder = HardCodedUnetAutoencoder()
    hard_coded_unet_autoencoder.test_hard_coded_unet(df)

if __name__ == "__main__":
    main()