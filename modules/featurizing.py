import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class TransformerFitter(TransformerMixin):
    def fit(self, X=None, y=None):
        return self

class Featurizer(TransformerFitter):

    def __init__(self, houses):
        self.houses = houses

    def transform(self, houses, codes, services, infrastructure, leisure):


        # dummy code categoryObject
        houses = pd.get_dummies(houses, columns = ['categoryObject'])

        # encode energy labels 
        houses['energyLabel'] = houses['energyLabel'].fillna(0)
        houses['energyLabel'].replace('A+++++', 1, inplace=True)
        houses['energyLabel'].replace('A++++', 2, inplace=True)
        houses['energyLabel'].replace('A+++', 3, inplace=True)
        houses['energyLabel'].replace('A++', 4, inplace=True)
        houses['energyLabel'].replace('A+', 5, inplace=True)
        houses['energyLabel'].replace('A', 6, inplace=True)
        houses['energyLabel'].replace('B', 7, inplace=True)
        houses['energyLabel'].replace('C', 8, inplace=True)
        houses['energyLabel'].replace('D', 9, inplace=True)
        houses['energyLabel'].replace('E', 10, inplace=True)
        houses['energyLabel'].replace('F', 11, inplace=True)
        houses['energyLabel'].replace('G', 12, inplace=True)

        # replace missing values houseTypes
        houses['houseType1'] = houses['houseType1'].fillna(0)
        # dummy code houseType1
        houses = pd.get_dummies(houses, columns = ['houseType1'])

        # replace missing values houses types
        houses['houseType2'] = houses['houseType2'].fillna(0)
        # dummy code houseType2
        houses = pd.get_dummies(houses, columns = ['houseType2'])

        #Calculate sellingtime
        houses['signDate'] = pd.to_datetime(houses['signDate'])
        houses['publicationDate'] = pd.to_datetime(houses['publicationDate'])
        houses['sellingtime'] = houses['signDate'] - houses['publicationDate']
        houses['sellingtime'] = houses['sellingtime'].dt.days.astype('int16')

        #Calculate object age
        houses['constructionYear'] = pd.to_numeric(houses['constructionYear'])
        houses['objectAge'] = 2018 - houses['constructionYear']

        #Merge houses with codes
        houses_codes = pd.merge(houses, codes, how='left', left_on='zipcode', right_on='PC6')
        #Remove data records that did not match with codes
        houses_codes = houses_codes[houses_codes['PC6'].notna()]

        #### merge feature_codes with services 
        houses_codes_services = pd.merge(houses_codes, services, how='left', left_on='Buurt2018', right_on='Codering_3')

        #### merge feature_codes_services with infrastructure
        houses_codes_services_infrastructure = pd.merge(houses_codes_services, infrastructure, how='left', left_on='Codering_3', right_on='coding')

        #### merge houses_codes_services_infrastructure with leisure
        houses_cbs = pd.merge(houses_codes_services_infrastructure, leisure, how='left', left_on='Codering_3', right_on='coding')

        # removing letters from zipcode 
        houses_cbs['zipcode'] = [i[:4] for i in houses_cbs['zipcode']]
        # change to numeric value
        houses_cbs['zipcode'] = pd.to_numeric(houses_cbs['zipcode'])

        #Deleting after joining 3 CBS tables
        del houses_cbs['PC6']
        del houses_cbs['Buurt2018']
        del houses_cbs['Wijk2018']
        del houses_cbs['Gemeente2018']
        del houses_cbs['ID']
        del houses_cbs['Gemeentenaam_1']
        del houses_cbs['SoortRegio_2']
        del houses_cbs['Codering_3']
        del houses_cbs['Regioaanduiding/Gemeentenaam (naam)_x']
        del houses_cbs['Regioaanduiding/Soort regio (omschrijving)_x']
        del houses_cbs['Regioaanduiding/Codering (code)_x']
        del houses_cbs['coding_x']
        del houses_cbs['Regioaanduiding/Gemeentenaam (naam)_y']
        del houses_cbs['Regioaanduiding/Soort regio (omschrijving)_y']
        del houses_cbs['Regioaanduiding/Codering (code)_y']
        del houses_cbs['coding_y']
        del houses_cbs['publicationDate']
        del houses_cbs['description']
        del houses_cbs['office']
        del houses_cbs['signDate']
        del houses_cbs['constructionYear']
        del houses_cbs['zipcode_numbers']
        del houses_cbs['houseType']








        # #encode woningtype1        
        # houses['houseType1'].replace('woonboot', 0, inplace=True)
        # houses['houseType1'].replace('eengezinswoning', 1, inplace=True)
        # houses['houseType1'].replace('tussenverdieping', 2, inplace=True)
        # houses['houseType1'].replace('woonboerderij', 3, inplace=True)
        # houses['houseType1'].replace('portiekflat', 4, inplace=True)
        # houses['houseType1'].replace('galerijflat', 5, inplace=True)
        # houses['houseType1'].replace('villa', 6, inplace=True)
        # houses['houseType1'].replace('beneden + bovenwoning', 7, inplace=True)
        # houses['houseType1'].replace('bovenwoning', 8, inplace=True)
        # houses['houseType1'].replace('benedenwoning', 9, inplace=True)
        # houses['houseType1'].replace('maisonnette', 10, inplace=True)
        # houses['houseType1'].replace('herenhuis', 11, inplace=True)
        # houses['houseType1'].replace('penthouse', 12, inplace=True)
        # houses['houseType1'].replace('bungalow', 13, inplace=True)
        # houses['houseType1'].replace('landhuis', 14, inplace=True)
        # houses['houseType1'].replace('grachtenpand', 15, inplace=True)
        # houses['houseType1'].replace('dubbel benedenhuis', 16, inplace=True)
        # houses['houseType1'].replace('landgoed', 17, inplace=True)
        # houses['houseType1'].replace('studentenkamer', 18, inplace=True)
        # houses['houseType1'].replace('stacaravan', 19, inplace=True)
        # houses['houseType1'].replace('portiekwoning', 20, inplace=True)

        # #Calculate object age
        # houses['constructionYear'] = pd.to_numeric(houses['constructionYear'])
        # houses['objectAge'] = 2018 - houses['constructionYear']
        # #houses['constructionYear'] = [i[:4] for i in houses['constructionYear']]

        # #Calculate sellingtime
        # houses['publicationDate'] = pd.to_datetime(houses['publicationDate'])
        # houses['signDate'] = pd.to_datetime(houses['signDate'])

        # houses['sellingtime'] = houses['signDate'] - houses['publicationDate']
        # houses['sellingtime'] = houses['sellingtime'].dt.days.astype('int16')

        # #Removing columns that are not needed
        # #del houses['globalId']
        # del houses['publicationDate']
        # del houses['zipcode']
        # del houses['description']
        # del houses['office']
        # del houses['signDate']
        # del houses['houseType2']
        # del houses['constructionYear']
        # del houses['zipcode_numbers']
        # del houses['houseType'] # this one is encoded --> redundant
        # del houses['parcel']


        #From here and below is included in cleaning and may need to be moved
        # categorieObject 0 = woonhuis, 1 = appartement
        #houses['categoryObject'].replace('<{Woonhuis}>', 0, inplace=True)
        #houses['categoryObject'].replace('<{Appartement}>', 1, inplace=True)

        # split houseType into houseType1 and houseType2, drop houseType
        #houses[['houseType1','houseType2']] = houses.houseType.str.split("}>",1,expand=True)
        #houses['houseType1'] = houses['houseType1'].str[2:]


        #FOR NIELS: we should discuss how to do this
        
        #final_houses = final_houses.merge(houses, how='left')
        houses = houses_cbs

        return houses

# self = Featurizer(houses = houses)
