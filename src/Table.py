import pandas as pd


class Table:
    def __init__(self, data_frame: pd.DataFrame):
        self.table = data_frame

    def add_column(self, name, data=None):
        if data is None:
            self.table[name] = ""

        self.table[name] = data
        return self.table

    def save_csv(self, path: str, index=False):
        self.table.to_csv(path, index=index)
