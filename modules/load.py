import os
import pandas as pd

class DataLoader(object):
    def __init__(self, raw_folder):
        self.raw_folder=raw_folder

    def load_data(self):
        houses = pd.read_csv(os.path.join(self.raw_folder, 'housing_data.csv'))
        codes = pd.read_csv(os.path.join(self.raw_folder, 'codes.csv'),sep=";")
        services = pd.read_csv(os.path.join(self.raw_folder, 'services.csv'),sep=";")
        infrastructure = pd.read_csv(os.path.join(self.raw_folder, 'infrastructure.csv'),sep=";")
        leisure = pd.read_csv(os.path.join(self.raw_folder, 'leisure.csv'),sep=";")
        return houses, codes, services, infrastructure, leisure