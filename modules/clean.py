import pandas as pd
import numpy as np

class DataCleaner(object):

    def __init__(self):
        self.cleaning_statistics = {}

    def clean_data(self, houses, codes, services, infrastructure, leisure) -> list:

        ##### CLEAN HOUSES ########
        
        # delete globalId.1 and change columns names
        del houses['globalId.1']
        
        houses.columns = ['globalId', 'publicationDate', 'zipcode', 'sellingPrice', 'description', 'houseType', 
        'categoryObject', 'constructionYear', 'garden', 'parcel', 'office', 'rooms', 'bathrooms', 
        'energyLabel', 'livingArea', 'signDate']

        # replace nan
        houses['bathrooms'] = houses['bathrooms'].replace(np.NaN, houses["bathrooms"].mean())
        houses['sellingPrice'] = houses['sellingPrice'].replace(np.NaN, houses["sellingPrice"].mean())

        #Round # of bathrooms
        houses['bathrooms'] = houses['bathrooms'].round(decimals=0)

        # set threshold house price 50000
        houses = houses.drop(houses[houses['sellingPrice']<=50000].index)

        #remove outlier oppervlakte (1)
        meansurface = houses['livingArea'].mean()
        houses['livingArea'].replace(1, meansurface, inplace=True)

        # remove part of the output
        houses['constructionYear'] = houses['constructionYear'].str[-4:]
        houses['zipcode_numbers'] = [i[:4] for i in houses['zipcode']] #store in new column

        # categoryObject 0 = woonhuis, 1 = appartement
        houses['categoryObject'].replace('<{Woonhuis}>', 0, inplace=True)
        houses['categoryObject'].replace('<{Appartement}>', 1, inplace=True)

        # split soortWoning and encode soortwoning1
        houses[['houseType1','houseType2']] = houses.houseType.str.split("}>",1,expand=True)
        houses['houseType1'] = houses['houseType1'].str[2:]

        # change dtype
        houses['publicationDate'] = pd.to_datetime(houses['publicationDate'])
        houses['signDate'] = pd.to_datetime(houses['signDate'])

        # fill parcel with livingarea when nan
        houses.parcel.fillna(houses.livingArea, inplace = True)


        ######## CLEAN CODES ############

        # Deleting the collumn huisnummer
        codes = codes.drop(columns="Huisnummer")

        # sorting by first name
        codes.sort_values("PC6", inplace = True)

        # dropping ALL duplicte values
        codes.drop_duplicates(subset=["PC6"], keep = 'last', inplace = True)

        ######## CLEAN SERVICES ############

        #trim the spaces for column AfstandTotGroteSupermarkt_96
        services['AfstandTotGroteSupermarkt_96'] = services['AfstandTotGroteSupermarkt_96'].str.strip()
        #replace the '.' for blank values in column AfstandTotGroteSupermarkt_96
        services['AfstandTotGroteSupermarkt_96'] = services['AfstandTotGroteSupermarkt_96'].replace(['.'], np.nan)
        #trim the spaces for column AfstandTotSchool_98
        services['AfstandTotSchool_98'] = services['AfstandTotSchool_98'].str.strip()
        #replace the '.' for blank values in column AfstandTotSchool_98
        services['AfstandTotSchool_98'] = services['AfstandTotSchool_98'].replace(['.'], np.nan)
        #trim the spaces for column ScholenBinnen3Km_99
        services['ScholenBinnen3Km_99'] = services['ScholenBinnen3Km_99'].str.strip()
        #replace the '.' for blank values in column ScholenBinnen3Km_99
        services['ScholenBinnen3Km_99'] = services['ScholenBinnen3Km_99'].replace(['.'], np.nan)
        #delete first two characters for all rows in column Codering_3
        services['Codering_3'] = [i[2:] for i in services['Codering_3']]
        #delete column WijkenEnBuurten
        del services['WijkenEnBuurten']
        #change datatype from object to numeric for Codering_3
        services['Codering_3'] = pd.to_numeric(services['Codering_3'])
        #change datatype from object to numeric for AfstandTotGroteSupermarkt_96
        services['AfstandTotGroteSupermarkt_96'] = pd.to_numeric(services['AfstandTotGroteSupermarkt_96'])
        #change datatype from object to numeric for AfstandTotSchool_98
        services['AfstandTotSchool_98'] = pd.to_numeric(services['AfstandTotSchool_98'])
        #change datatype from object to numeric for ScholenBinnen3Km_99
        services['ScholenBinnen3Km_99'] = pd.to_numeric(services['ScholenBinnen3Km_99'])
        #trim spaces gemeentenaam
        services['Gemeentenaam_1'] = services['Gemeentenaam_1'].str.strip()
        #trim spaces soortregio
        services['SoortRegio_2'] = services['SoortRegio_2'].str.strip()

        # replace missing values with mean of distinct column
        services['AfstandTotHuisartsenpraktijk_95'] = services['AfstandTotHuisartsenpraktijk_95'].replace(np.NaN, services['AfstandTotHuisartsenpraktijk_95'].mean())

        services['AfstandTotGroteSupermarkt_96'] = services['AfstandTotGroteSupermarkt_96'].replace(np.NaN, services['AfstandTotGroteSupermarkt_96'].mean())

        services['AfstandTotKinderdagverblijf_97'] = services['AfstandTotKinderdagverblijf_97'].replace(np.NaN, services['AfstandTotKinderdagverblijf_97'].mean())

        services['AfstandTotSchool_98'] = services['AfstandTotSchool_98'].replace(np.NaN, services['AfstandTotSchool_98'].mean())

        services['ScholenBinnen3Km_99'] = services['ScholenBinnen3Km_99'].replace(np.NaN, services['ScholenBinnen3Km_99'].mean())


        ######### CLEAN INFRASTRUCTURE ###########
        infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'] = infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'].str.strip()
        infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'] = infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'].replace(['.'], np.nan)

        infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'] = infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'].str.strip()
        infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'] = infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'].replace(['.'], np.nan)

        infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'] = infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'].str.strip()
        infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'] = infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'].replace(['.'], np.nan)

        infrastructure['Regioaanduiding/Gemeentenaam (naam)'] = infrastructure['Regioaanduiding/Gemeentenaam (naam)'].str.strip()

        infrastructure['Regioaanduiding/Soort regio (omschrijving)'] = infrastructure['Regioaanduiding/Soort regio (omschrijving)'].str.strip()

        infrastructure['Regioaanduiding/Codering (code)'] = infrastructure['Regioaanduiding/Codering (code)'].str.strip()

        infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'] = pd.to_numeric(infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'])
        infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'] = pd.to_numeric(infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'])
        infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'] = pd.to_numeric(infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'])

        infrastructure['coding'] = infrastructure['Regioaanduiding/Codering (code)']
        infrastructure['coding'] = [i[2:] for i in infrastructure['coding']]

        #belowd added to be able to merge float on float
        infrastructure['coding'] = infrastructure.coding.astype(float)

        del infrastructure['Wijken en buurten']

        # replace missing values with mean of distinct column
        infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'] = infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'].replace(np.NaN, infrastructure['Verkeer en vervoer/Afstand tot oprit hoofdverkeersweg (km)'].mean())

        infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'] = infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'].replace(np.NaN, infrastructure['Verkeer en vervoer/Treinstations/Afstand tot treinstations totaal (km)'].mean())

        infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'] = infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'].replace(np.NaN, infrastructure['Verkeer en vervoer/Treinstations/Afstand tot belangrijk overstapstation (km)'].mean())



        ######## CLEAN LEISURE ###########
        leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'] = leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'].str.strip()
        leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'] = leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'].replace(['.'], np.nan)

        leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'] = leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'].str.strip()
        leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'] = leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'].replace(['.'], np.nan)

        leisure['Horeca/Restaurants/Afstand tot restaurant (km)'] = leisure['Horeca/Restaurants/Afstand tot restaurant (km)'].str.strip()
        leisure['Horeca/Restaurants/Afstand tot restaurant (km)'] = leisure['Horeca/Restaurants/Afstand tot restaurant (km)'].replace(['.'], np.nan)

        leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'] = leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'].str.strip()
        leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'] = leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'].replace(['.'], np.nan)

        leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'] = leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'].str.strip()
        leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'] = leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'].replace(['.'], np.nan)

        leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'] = leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'].str.strip()
        leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'] = leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'].replace(['.'], np.nan)

        leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'] = leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'].str.strip()
        leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'] = leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'].replace(['.'], np.nan)

        leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'] = leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'].str.strip()
        leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'] = leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'].replace(['.'], np.nan)

        leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'] = leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'].str.strip()
        leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'] = leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'].replace(['.'], np.nan)

        leisure['Regioaanduiding/Gemeentenaam (naam)'] = leisure['Regioaanduiding/Gemeentenaam (naam)'].str.strip()

        leisure['Regioaanduiding/Soort regio (omschrijving)'] = leisure['Regioaanduiding/Soort regio (omschrijving)'].str.strip()

        leisure['Regioaanduiding/Codering (code)'] = leisure['Regioaanduiding/Codering (code)'].str.strip()

        del leisure['Wijken en buurten']

        leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'] = pd.to_numeric(leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'])

        leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'] = pd.to_numeric(leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'])

        leisure['Horeca/Restaurants/Afstand tot restaurant (km)'] = pd.to_numeric(leisure['Horeca/Restaurants/Afstand tot restaurant (km)'])

        leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'] = pd.to_numeric(leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'])

        leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'] = pd.to_numeric(leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'])

        leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'] = pd.to_numeric(leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'])

        leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'] = pd.to_numeric(leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'])

        leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'] = pd.to_numeric(leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'])

        leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'] = pd.to_numeric(leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'])

        leisure['coding'] = leisure['Regioaanduiding/Codering (code)']
        leisure['coding'] = [i[2:] for i in leisure['coding']]
        leisure['coding'] = pd.to_numeric(leisure['coding'])

        # replace missing values with mean of distinct column
        leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'] = leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'].replace(np.NaN, leisure['Horeca/Cafés en dergelijke/Afstand tot café e.d. (km)'].mean())

        leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'] = leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'].replace(np.NaN, leisure['Horeca/Cafés en dergelijke/Aantal cafés/Binnen 3 km (aantal)'].mean())

        leisure['Horeca/Restaurants/Afstand tot restaurant (km)'] = leisure['Horeca/Restaurants/Afstand tot restaurant (km)'].replace(np.NaN, leisure['Horeca/Restaurants/Afstand tot restaurant (km)'].mean())

        leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'] = leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'].replace(np.NaN, leisure['Horeca/Restaurants/Aantal restaurants/Binnen 3 km (aantal)'].mean())

        leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'] = leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'].replace(np.NaN, leisure['Vrije tijd en cultuur/Afstand tot bibliotheek (km)'].mean())

        leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'] = leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'].replace(np.NaN, leisure['Vrije tijd en cultuur/Museum/Afstand tot museum (km)'].mean())

        leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'] = leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'].replace(np.NaN, leisure['Vrije tijd en cultuur/Museum/Aantal musea/Binnen 20 km (aantal)'].mean())

        leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'] = leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'].replace(np.NaN, leisure['Vrije tijd en cultuur/Bioscoop/Afstand tot bioscoop (km)'].mean())

        leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'] = leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'].replace(np.NaN, leisure['Vrije tijd en cultuur/Bioscoop/Aantal bioscopen/Binnen 20 km (aantal)'].mean())


        return houses, codes, services, infrastructure, leisure
