#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np

mortgages_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Test_mortgages.csv'
incentives_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Incentives.csv'
# Lees de CSV-bestanden in
mortgages_df = pd.read_csv(mortgages_path, delimiter=',', encoding='ISO-8859-1')
incentives_df = pd.read_csv(incentives_path, delimiter=',', encoding='ISO-8859-1')

print(incentives_df.columns)


# In[17]:


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


# In[18]:


def calculate_aging(mortgages_df, spot_date):
    # Bereken de leeftijd van de hypotheek in maanden
    spot_date = pd.to_datetime(spot_date)
    mortgages_df['age'] = (spot_date.year - mortgages_df['Datum Ingang Leningdeel'].dt.year) * 12 + \
                          (spot_date.month - mortgages_df['Datum Ingang Leningdeel'].dt.month)
    return mortgages_df

# Bereken de leeftijd van de hypotheken
spot_date = '2023-03-31'
mortgages_df = calculate_aging(mortgages_df, spot_date)

# Resultaten tonen
#print(mortgages_df[['Datum Ingang Leningdeel', 'age']].head(20))


# In[19]:


def determine_segment(row):
    if 'Fixed' in row and row['Fixed'] == 1:
        return 10
    elif 'age' in row and row['age'] == 1:
        return 8
    elif row['Aflosvorm'].startswith("Belegging"):
        return 7
    elif row['Aflosvorm'].startswith("Leven"):
        return 5
    elif row['Aflosvorm'].startswith("Spaar"):
        return 5
    elif row['Aflosvorm'].startswith("Annuit"):
        return 1
    elif row['Aflosvorm'].startswith("Lineair"):
        return 2
    elif row['Aflosvorm'] == "Aflossingsvrij" and row['LTV Unindexed'] < 1:
        return 3
    elif row['Aflosvorm'] == "Aflossingsvrij" and row['LTV Unindexed'] >= 1:
        return 4
    else:
        return 9

# Voeg placeholders toe voor de kolom 'Fixed' als deze niet bestaat
if 'Fixed' not in mortgages_df.columns:
    mortgages_df['Fixed'] = 0  # Of gebruik een andere logische waarde

# Bereken het segment
mortgages_df['Segment'] = mortgages_df.apply(determine_segment, axis=1)

# Toon de eerste paar rijen van de DataFrame om te controleren
relevant_columns = ['Aflosvorm', 'LTV Unindexed', 'Datum Ingang RVP', 'age', 'Segment']
print(mortgages_df[relevant_columns].head(20))


# In[12]:


def calculate_seasonality(incentives_df, segment):
    seasonality_rows = incentives_df[incentives_df['ParameterClass'] == 'Parameter']
    seasonality_rows = seasonality_rows[seasonality_rows['Coefficient'].str.startswith('season_')]
    
    seasonality_dict = seasonality_rows.set_index('Coefficient')[segment].to_dict()
    return seasonality_dict
# Testen van de calculate_seasonality functie
test_segment = 'Segment 1a'
seasonality_dict = calculate_seasonality(incentives_df, test_segment)
print(f'Seasonality factors for {test_segment}:')
print(seasonality_dict)


# In[13]:


def apply_seasonality_to_mortgages(mortgages_df, incentives_df, spot_date):
    spot_date = pd.to_datetime(spot_date)
    
    # Stel een array van maanden in vanaf de startdatum
    months = pd.date_range(start=spot_date, periods=12, freq='M')
    results = pd.DataFrame(index=mortgages_df.index)

    for index, row in mortgages_df.iterrows():
        segment = row['Segment']
        seasonality_dict = calculate_seasonality(incentives_df, segment)
        
        for i, month in enumerate(months):
            seasonality_factor = seasonality_dict.get(f'season_{


# In[ ]:




