#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


mortgages_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Test_mortgages.csv'
incentives_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Incentives.csv'
mapping_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/prepayment_mapping.csv'

# Lees de CSV-bestanden in
mortgages_df = pd.read_csv(mortgages_path, delimiter=',', encoding='ISO-8859-1')
incentives_df = pd.read_csv(incentives_path, delimiter=',', encoding='ISO-8859-1')
mapping_df = pd.read_csv(mapping_path)

#print(incentives_df.columns)
#print(incentives_df.head(25))


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


# In[9]:


def calculate_aging(mortgages_df, spot_date):
    # Omzetten naar datetime
    spot_date = pd.to_datetime(spot_date)
    
    # Bereken de leeftijd van de hypotheek in fractionele jaren
    mortgages_df['Datum Ingang Leningdeel'] = pd.to_datetime(mortgages_df['Datum Ingang Leningdeel'])
    mortgages_df['Mortgage_age'] = (spot_date - mortgages_df['Datum Ingang Leningdeel']).dt.days / 365.25  # 365.25 om rekening te houden met schrikkeljaren
    
    return mortgages_df

# Bereken de leeftijd van de hypotheken
spot_date = '2023-03-31'
mortgages_df = calculate_aging(mortgages_df, spot_date)

# Resultaten tonen
print(mortgages_df[['Datum Ingang Leningdeel', 'Mortgage_age']].head(20))


# In[10]:


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


# In[12]:


age_parameters = incentives_df[incentives_df['Coefficient'].isin(['e_age', 'f_age'])]
age_parameters = age_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]
age_parameters = age_parameters.set_index('Coefficient').apply(pd.to_numeric)

# Functie om de leeftijdsfactor voor een gegeven rij te berekenen
def calculate_age_factor(row):
    segment = row['Segment']
    e_age = age_parameters.loc['e_age', segment]
    f_age = age_parameters.loc['f_age', segment]
    age = row['Mortgage_age']
    age_factor = min(1, e_age + f_age * age)
    return age_factor

# Pas de functie toe op elke rij in de DataFrame om de leeftijdsfactor te berekenen
mortgages_df['age'] = mortgages_df.apply(calculate_age_factor, axis=1)


# In[14]:


print(mortgages_df[['Aflosvorm', 'LTV Unindexed', 'Datum Ingang Leningdeel', 'Mortgage_age', 'Segment', 'age']])


# In[15]:


# Filter de seizoensfactoren
seasonality_parameters = incentives_df[incentives_df['Coefficient'].str.startswith('season_')]
# Selecteer alleen de kolommen tot en met segment 9
seasonality_parameters = seasonality_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]
#print(seasonality_parameters)
#print(seasonality_parameters.columns)


# In[16]:


def create_seasonality_matrix(mortgages_df, spot_date, seasonality_parameters):
    spot_date = pd.to_datetime(spot_date)
    n_months = 360
    seasonality_matrices = {}

    for i, row in mortgages_df.iterrows():
        segment = row['Segment']
        # Verwijder 'Segment' en haal het nummer op
        segment_index = segment.replace('Segment ', '')
        segment_col = f'Segment {segment_index}'
        
        # Haal de seizoensfactoren voor het specifieke segment op
        season_factors = seasonality_parameters[segment_col].values
        
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


# In[ ]:




