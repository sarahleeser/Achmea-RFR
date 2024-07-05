#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


mortgages_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Test_mortgages.csv'
incentives_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Incentives.csv'
mapping_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/prepayment_mapping.csv'
rates_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Mortgagerates.csv'

# Lees de CSV-bestanden in
mortgages_df = pd.read_csv(mortgages_path, delimiter=',', encoding='ISO-8859-1')
incentives_df = pd.read_csv(incentives_path, delimiter=',', encoding='ISO-8859-1')
rates_df = pd.read_csv(rates_path, delimiter=',', encoding='ISO-8859-1')
mapping_df = pd.read_csv(mapping_path)

#print(incentives_df.columns)
#print(incentives_df.head(25))

print(mortgages_df.head())
print(rates_df.head())


# In[3]:


mortgages_df = mortgages_df.merge(mapping_df, left_on='Aflosvorm', right_on='AflossingsvormOmschrijving', how='left')

# Print de samengevoegde DataFrame om te controleren of de PrepaymentType correct is toegevoegd
#print(mortgages_df.head(20))


# In[4]:


# Filter de seizoensfactoren
seasonality_parameters = incentives_df[incentives_df['Coefficient'].str.startswith('season_')]
# Selecteer alleen de kolommen tot en met segment 9
seasonality_parameters = seasonality_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]
print(seasonality_parameters)
print(seasonality_parameters.columns)


# In[5]:


def parse_dates(date_str):
    # Controleer of de datum niet al als datetime-object is herkend
    if isinstance(date_str, pd.Timestamp):
        return date_str
    # Probeer de datum te parseren in verschillende formaten
    for fmt in ('%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%d-%m-%Y'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    # Als geen enkel formaat werkt, retourneer NaT
    return pd.NaT

# Pas de aangepaste datumparser toe
mortgages_df['Datum Ingang Leningdeel'] = mortgages_df['Datum Ingang Leningdeel'].apply(parse_dates)

# Controleer of de datums correct zijn geparsed
#print(mortgages_df['Datum Ingang Leningdeel'].head(20))


# In[6]:


def calculate_aging(mortgages_df, spot_date):
    # Omzetten naar datetime
    spot_date = pd.to_datetime(spot_date)
    
    # Bereken de leeftijd van de hypotheek in fractionele jaren
    mortgages_df['Datum Ingang Leningdeel'] = pd.to_datetime(mortgages_df['Datum Ingang Leningdeel'])
    mortgages_df['Mortgage_age'] = (spot_date - mortgages_df['Datum Ingang Leningdeel']).dt.days / 365.25  # 365.25 om rekening te houden met schrikkeljaren
    
    # Definieer de lengte van de leeftijdsmatrix
    n_months = 360
    aging_matrices = {}  # Dictionary om de leeftijdsmatrices op te slaan

    for i, row in mortgages_df.iterrows():
        start_age = row['Mortgage_age']
        rvp_end_date = pd.to_datetime(row['Datum Eind RVP'])
        months_active = min((rvp_end_date.year - spot_date.year) * 12 + rvp_end_date.month - spot_date.month, n_months)

        # Maak een array om de leeftijd per maand op te slaan
        aging_matrix = np.zeros(n_months)

        for month in range(months_active):
            aging_matrix[month] = start_age + (month + 1) / 12.0

        # Sla de leeftijdsmatrix op in de dictionary
        aging_matrices[f'Hypotheek {i}'] = aging_matrix

    return mortgages_df, aging_matrices

# Bereken de leeftijd van de hypotheken en genereer de leeftijdsmatrix
spot_date = '2023-03-31'
mortgages_df, aging_matrices = calculate_aging(mortgages_df, spot_date)

# Toon de resultaten voor de eerste paar hypotheken
#print("Voorbeeldresultaten:")
#print(mortgages_df[['Datum Ingang Leningdeel', 'Mortgage_age']].head(20))
# Controleer de matrix voor alle hypotheken
for i in range(len(mortgages_df)):
    print(f"Leeftijdsmatrix voor hypotheek {i}:")
    print(aging_matrices[f'Hypotheek {i}'])


# In[7]:


def determine_segment(row):
    # Specifieke logica voor bridge loans en Acier
    if 'ProductLijn' in row and 'Overbrugging' in row['ProductLijn']:
        row['PrepaymentType'] = 'bridge loans'
    if 'Merk' in row and row['Merk'] in ['Staal euro lening of RC', 'Acier']:
        row['PrepaymentType'] = 'Acier'

    # Bepaal het segment op basis van PrepaymentType, LTV Unindexed en age
    if row['PrepaymentType'] == 'annuity':
        return 'Segment 1a'
    elif row['PrepaymentType'] == 'linear':
        return 'Segment 1b'
    elif row['PrepaymentType'] == 'bullet' and row['LTV Unindexed'] < 1:
        return 'Segment 2'
    elif row['PrepaymentType'] == 'bullet' and row['LTV Unindexed'] >= 1:
        return 'Segment 3'
    elif row['PrepaymentType'] == 'savings' and row['Mortgage_age'] < 35:
        return 'Segment 4'
    elif row['PrepaymentType'] == 'savings' and row['Mortgage_age'] >= 35:
        return 'Segment 5'
    elif row['PrepaymentType'] == 'investment':
        return 'Segment 6'
    elif row['PrepaymentType'] == 'bridge loans':
        return 'Segment 7'
    elif row['PrepaymentType'] == 'other':
        return 'Segment 8'
    else:
        return 'Segment 9'  # Acier

# Voeg placeholders toe voor de kolom 'Fixed' als deze niet bestaat
if 'Fixed' not in mortgages_df.columns:
    mortgages_df['Fixed'] = 0  # Of gebruik een andere logische waarde

# Bereken het segment
mortgages_df['Segment'] = mortgages_df.apply(determine_segment, axis=1)


# In[8]:


age_parameters = incentives_df[incentives_df['Coefficient'].isin(['e_age', 'f_age'])]
age_parameters = age_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]
age_parameters = age_parameters.set_index('Coefficient').apply(pd.to_numeric)

# Functie om de leeftijdsfactor voor een gegeven leeftijd te berekenen
def calculate_age_factor(age, e_age, f_age):
    return min(1, e_age + f_age * age)

# Genereer de age factor matrices
age_factor_matrices = {}

for i, row in mortgages_df.iterrows():
    segment = row['Segment']
    e_age = age_parameters.loc['e_age', segment]
    f_age = age_parameters.loc['f_age', segment]
    aging_matrix = aging_matrices[f'Hypotheek {i}']
    age_factor_matrix = np.zeros_like(aging_matrix)
    
    for month in range(len(aging_matrix)):
        age_factor_matrix[month] = calculate_age_factor(aging_matrix[month], e_age, f_age)
    
    rvp_end_date = pd.to_datetime(row['Datum Eind RVP'])
    rvp_index = min(int((rvp_end_date - pd.to_datetime(spot_date)).days / 30.44), len(aging_matrix))
    
    # Set values after RVP to zero
    if rvp_index < len(aging_matrix):
        age_factor_matrix[rvp_index:] = 0
    
    age_factor_matrices[f'Hypotheek {i}'] = age_factor_matrix

# Toon de age factor matrices voor de eerste paar hypotheken
for i in range(len(mortgages_df)):
    print(f"Age factor matrix voor hypotheek {i}:")
    print(age_factor_matrices[f'Hypotheek {i}'])


# In[9]:


print(mortgages_df[['Aflosvorm', 'LTV Unindexed', 'Datum Ingang Leningdeel', 'Mortgage_age', 'Segment']])


# In[10]:


# Filter de seizoensfactoren
seasonality_parameters = incentives_df[incentives_df['Coefficient'].str.startswith('season_')]
# Selecteer alleen de kolommen tot en met segment 9
seasonality_parameters = seasonality_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]
#print(seasonality_parameters)
#print(seasonality_parameters.columns)


# In[51]:


def create_seasonality_matrix(mortgages_df, spot_date, seasonality_parameters):
    spot_date = pd.to_datetime(spot_date)
    n_months = 360
    seasonality_matrices = {}

    for i, row in mortgages_df.iterrows():
        segment = row['Segment']
        # Verwijder 'Segment' en haal het nummer op
        segment_index = segment.replace('Segment ', '')
        segment_col = f'Segment {segment_index}'
        
        # Haal de seizoensfactoren voor het specifieke segment op en converteer naar numerieke waarden
        season_factors = pd.to_numeric(seasonality_parameters[segment_col].values)
        
        # Verhoog elke factor met 1
        season_factors = season_factors + 1
        
        rvp_end_date = pd.to_datetime(row['Datum Eind RVP'])
        months_active = min((rvp_end_date.year - spot_date.year) * 12 + rvp_end_date.month - spot_date.month, n_months)

        seasonality_matrix = np.zeros(n_months)

        start_month = (spot_date.month % 12)
        for month in range(1, months_active + 1):
            month_index = (start_month + month - 1) % 12
            seasonality_matrix[month - 1] = season_factors[month_index]

        seasonality_matrices[f'Hypotheek {i}'] = seasonality_matrix

    return seasonality_matrices

# Pas de seasonality matrix functie toe
seasonality_matrices = create_seasonality_matrix(mortgages_df, spot_date, seasonality_parameters)

# Print de seasonality matrix voor elke hypotheek
for hypotheek, matrix in seasonality_matrices.items():
    print(f"{hypotheek}:")
    print(matrix)


# In[52]:


rente_nominaal = mortgages_df['Rente Nominaal']

# Bekijk de eerste paar rijen van de 'Rente Nominaal' kolom
print(rente_nominaal.head(25))


# In[53]:


spotdate = pd.to_datetime('2023-03-31')
mortgages_df['Datum Eind RVP'] = pd.to_datetime(mortgages_df['Datum Eind RVP'])

# Bereken de RENTE VASTPERIODE
mortgages_df['RENTEVASTPERIODE'] = np.ceil((mortgages_df['Datum Eind RVP'] - spotdate) / np.timedelta64(1, 'Y')).astype(int).astype(str) + ' jaar'
print(mortgages_df[['Datum Ingang Leningdeel', 'Datum Eind RVP', 'RENTEVASTPERIODE']].head(20))


# In[54]:


# Voeg een kolom voor Datum Eind RVP toe aan mortgages_df (dit moet al aanwezig zijn)
mortgages_df['Datum Eind RVP'] = pd.to_datetime(mortgages_df['Datum Eind RVP'], errors='coerce')

# Bereken de initiale remainingRVP in maanden vanaf de spotdate
mortgages_df['initial_remainingRVP'] = np.ceil((mortgages_df['Datum Eind RVP'] - spotdate) / np.timedelta64(1, 'M')).astype(int)

# Functie om de risicoklasse te bepalen op basis van LTV
def determine_risk_class(ltv):
    if ltv > 1.00:
        return 'boven 100'
    elif 0.95 <= ltv < 1.00:
        return 'tot 95'
    elif 0.90 <= ltv < 0.95:
        return 'tot 90'
    elif 0.85 <= ltv < 0.90:
        return 'tot 85'
    elif 0.80 <= ltv < 0.85:
        return 'tot 85'
    elif 0.70 <= ltv < 0.80:
        return 'tot 80'
    elif 0.60 <= ltv < 0.70:
        return 'tot 70'
    else:
        return 'tot 60'

# Mapping van aflosvormen tussen mortgages_df en rates_df
aflosvorm_mapping = {
    'Annuiteit': 'Annuïtair',
    'Lineair': 'Lineair',
    'Aflossingsvrij': 'Aflossingsvrij',
    'Levensverzekering': 'Aflossingsvrij',  # Aannemen dat Levensverzekering ook onder Aflossingsvrij valt
    'Spaarrekening': 'Aflossingsvrij'      # Aannemen dat Spaarrekening ook onder Aflossingsvrij valt
}

# Functie om de aflosvorm te normaliseren volgens de mapping
def normalize_aflosvorm(aflosvorm):
    return aflosvorm_mapping.get(aflosvorm, 'Aflossingsvrij')

# Functie om de RVP naar boven af te ronden naar de dichtstbijzijnde categorie
def round_up_rvp(rvp):
    rvp_categories = rates_df['RENTEVASTPERIODE'].unique()
    rvp_categories = sorted([int(cat.split()[0]) for cat in rvp_categories])
    
    for category in rvp_categories:
        if rvp <= category:
            return f"{category} jaar"
    return f"{rvp_categories[-1]} jaar"

# Functie om het gemiddelde rentepercentage op te halen
def get_average_rate(merk, aflosvorm, rvp, risk_class):
    relevant_rates = rates_df[(rates_df['namabd'] == merk) &
                              (rates_df['productvorm'] == aflosvorm) &
                              (rates_df['RENTEVASTPERIODE'] == rvp) &
                              (rates_df['RISICOKLASSE'].str.strip() == risk_class)]
    
    if not relevant_rates.empty:
        return relevant_rates.iloc[0]['Average of RENTE']
    
    # Als er geen exacte match is, rond de RVP naar boven af naar de dichtstbijzijnde categorie
    rounded_rvp = round_up_rvp(int(rvp.split()[0]))
    relevant_rates = rates_df[(rates_df['namabd'] == merk) &
                              (rates_df['productvorm'] == aflosvorm) &
                              (rates_df['RENTEVASTPERIODE'] == rounded_rvp) &
                              (rates_df['RISICOKLASSE'].str.strip() == risk_class)]
    
    if not relevant_rates.empty:
        return relevant_rates.iloc[0]['Average of RENTE']
    
    return np.nan

# Functie om de matrix te maken
def create_interest_rate_matrix(mortgages_df, months=360):
    n_hypotheken = len(mortgages_df)
    rate_matrix = np.zeros((n_hypotheken, months))
    
    for i, row in mortgages_df.iterrows():
        risk_class = determine_risk_class(row['LTV Unindexed'])
        aflosvorm = normalize_aflosvorm(row['Aflosvorm'])
        remaining_rvp = row['initial_remainingRVP']
        
        for month in range(months):
            if month < remaining_rvp:
                rvp_category = round_up_rvp((remaining_rvp - month) / 12)
                rate = get_average_rate(row['Merk'], aflosvorm, rvp_category, risk_class)
                rate_matrix[i, month] = rate if rate is not np.nan else 0
            else:
                rate_matrix[i, month] = 0
    
    return rate_matrix

# Maak de matrix
interest_rate_matrix = create_interest_rate_matrix(mortgages_df)

# Print de matrices per hypotheek
for idx, matrix in enumerate(interest_rate_matrix):
    print(f"Hypotheek {idx}:")
    print(matrix)
    print("\n")


# In[55]:


def create_nominal_rate_matrix(mortgages_df, months=360):
    n_hypotheken = len(mortgages_df)
    nominal_rate_matrix = np.zeros((n_hypotheken, months))
    
    for i, row in mortgages_df.iterrows():
        nominal_rate = row['Rente Nominaal']
        remaining_rvp = row['initial_remainingRVP']
        
        for month in range(months):
            if month < remaining_rvp:
                nominal_rate_matrix[i, month] = nominal_rate
            else:
                nominal_rate_matrix[i, month] = 0
    
    return nominal_rate_matrix

# Maak de nominale rentematrix
nominal_rate_matrix = create_nominal_rate_matrix(mortgages_df)

# Bereken de incentivematrix
incentive_matrix = nominal_rate_matrix - interest_rate_matrix

# Print de incentivematrices per hypotheek
for idx, matrix in enumerate(incentive_matrix):
    print(f"Incentive Hypotheek {idx}:")
    print(matrix)
    print("\n")


# In[ ]:





# In[ ]:





# In[ ]:




