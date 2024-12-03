import pandas as pd
import os
import ast
import re
import numpy as np
class DataPreprocess:
    def test_data_preprocess(self, train_test_data, dataset_name):
        self.train_test_data = train_test_data
        self.train_data = self.train_test_data.drop(columns=['meta.number', 'meta.time_epoch', 'meta.time', 'DATA.data', 'DATA.data.data', 'tls.handshake.random_time', 'label','eth._ws.expert.message', 'tls.handshake.ja3_full', 'tls.handshake.ja3', 'tls.handshake.ja3s_full', 'tls.handshake.ja3s', 'tcp.flags.str', 'tcp.port'])
        self.train_data = self.preprocess_train_data(self.train_data)

        self.train_data.to_csv(dataset_name, index=False)


    def is_list_string_format(self, s):
        return re.match(r"^\[.*\]$", str(s)) is not None

    def is_list_int_format(self, s):
        return re.match(r"\['\d{1,4}',(\s*'\d{1,4}',)*\s*'\d{1,4}'\]", str(s)) is not None

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
    def split_colon_separated_values(self, df, col_name):
        # Split the column into separate columns
        split_values = df[col_name].str.split(':', expand=True)

        # Rename the columns
        split_values.columns = [f"{col_name}.{i}" for i in range(split_values.shape[1])]

        # Concatenate the new columns with the original DataFrame
        df = pd.concat([df, split_values], axis=1)

        return df

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
                df[col] = df[col].fillna('0')
            elif np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].fillna(0)
            else:
                raise Exception("Not implemented type in replace_nans")
        return df

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
            df.loc[:, col] = df[col].apply(lambda x: self.convert_values(x, dtype))
        return df

    def string_list_to_columns(self, df, col):
        # Flache Liste aller Werte in der Spalte erstellen
        all_values = set(val for sublist in df[col].apply(ast.literal_eval) for val in sublist)

        # DataFrame für neue Spalten erstellen
        new_columns = pd.DataFrame()

        # Für jeden Wert eine neue Spalte erstellen und mit 1 oder 0 befüllen
        for value in all_values:
            new_columns[f"{col}.{value}"] = df[col].apply(lambda x: 1 if value in ast.literal_eval(x) else 0)

        # Original-DataFrame mit den neuen Spalten zusammenführen
        df = pd.concat([df, new_columns], axis=1)

        return df

    def ensure_list(self, x):
        if isinstance(x, str):
            try:
                value = ast.literal_eval(x)
                if not isinstance(value, list):
                    return [x]
                return value
            except:
                return [x]
        elif not isinstance(x, list):
            return [x]
        return x

    def int_list_to_columns(self, df, col):
        df[col] = df[col].apply(self.ensure_list)
        # Maximum der Länge der Listen in der Spalte ermitteln
        max_length = df[col].apply(lambda x: x.count(',') + 1).max()

        # Neue Spalten basierend auf der maximalen Länge erstellen
        new_cols = [f"{col}_{i + 1}" for i in range(max_length)]

        # Neue Spalten mit leeren Werten erstellen
        for new_col in new_cols:
            df[new_col] = "0"

        # Werte aus der Liste in die entsprechenden Spalten eintragen
        for index, row in df.iterrows():
            try:
                values = row[col]#ast.literal_eval(row[col])
                for i, value in enumerate(values):
                    try:
                        df.at[index, f"{col}_{i + 1}"] = str(int(value))
                    except Exception as e:
                        print(e)
            except Exception as e:
                print(e)

        return df
    def preprocess_train_data(self, df):
        df = df.dropna(axis=1, how='all')
        #https://datagy.io/pandas-fillna/
        df = self.cast_hexstr_to_int(df)
        df = self.replace_nans(df)

        # transfer list columns to on hot encoding
        list_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str)).all()]
        # type(df['sll.ltype'].iloc[14])
        # type(df['sll.ltype'].iloc[15])

        for col in list_columns:

            if df[col].apply(self.is_colon_separated_format).any():
                try:
                    df = self.split_colon_separated_values(df, col)
                except Exception as e:
                    print(f"Error in 1 for col: {col}. Error {e}")

            elif df[col].apply(self.is_abbreviated_ipv6_format).any():
                try:
                    df = self.split_ipv6_into_hextets(df, col)
                except Exception as e:
                    print(f"Error in 2 for col: {col}. Error {e}")

            elif df[col].apply(self.is_abbreviated_ipv6_list_format).any():
                try:
                    df = self.split_ipv6_list_into_hextets(df, col)
                except Exception as e:
                    print(f"Error in 3 for col: {col}. Error {e}")

            elif df[col].apply(self.is_ipv4_format).any():
                try:
                    df = self.split_ipv4_into_octets(df, col)
                except Exception as e:
                    print(f"Error in 4 for col: {col}. Error {e}")

            elif df[col].apply(self.is_list_int_format).any():
                try:
                    df = self.int_list_to_columns(df, col)
                except Exception as e:
                    print(f"Error in 5 for col: {col}. Error {e}")

            elif df[col].apply(self.is_list_string_format).any():
                try:
                    df = self.string_list_to_columns(df, col)
                except Exception as e:
                    print(f"Error in 5 for col: {col}. Error {e}")

            else:
                raise Exception("Not implemented converter in preprocess_train_data for col: {col}")

            # Originalspalte entfernen
            df.drop(columns=[col], inplace=True)

        return df


def main():
    data_name = 'csv-example'

    csv_path = os.path.join(os.path.dirname(__file__), '..', f"{data_name}.csv")

    sample_df = pd.read_csv(csv_path, low_memory=False)
    column_types = sample_df.dtypes.to_dict()
    del sample_df

    df = pd.read_csv(csv_path, dtype=column_types)

    data_preprocess = DataPreprocess()
    data_preprocess.test_data_preprocess(df, f"{data_name}_preprocessed.csv")

if __name__ == "__main__":
    main()