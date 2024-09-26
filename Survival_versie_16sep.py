#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle


# In[3]:


mortgages_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Test_mortgages.csv'
incentives_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Incentives.csv'
mapping_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/prepayment_mapping.csv'
rates_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Mortgagerates.csv'

# Lees de CSV-bestanden in
mortgages_df = pd.read_csv(mortgages_path, delimiter=',', encoding='ISO-8859-1')
incentives_df = pd.read_csv(incentives_path, delimiter=',', encoding='ISO-8859-1')
rates_df = pd.read_csv(rates_path, delimiter=',', encoding='ISO-8859-1')
mapping_df = pd.read_csv(mapping_path)

print(mortgages_df.head())
print(rates_df.head())


# In[4]:


mortgages_df = mortgages_df.merge(mapping_df, left_on='Aflosvorm', right_on='AflossingsvormOmschrijving', how='left')

# In[20]:
# Filter de seizoensfactoren
seasonality_parameters = incentives_df[incentives_df['Coefficient'].str.startswith('season_')]
seasonality_parameters = seasonality_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]
print(seasonality_parameters)
print(seasonality_parameters.columns)


# In[5]:


def parse_dates(date_str):
    if isinstance(date_str, pd.Timestamp):
        return date_str
    for fmt in ('%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%d-%m-%Y'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

# Pas de aangepaste datumparser toe
mortgages_df['Datum Ingang Leningdeel'] = mortgages_df['Datum Ingang Leningdeel'].apply(parse_dates)


# In[6]:


def calculate_aging(mortgages_df, spot_date):
    spot_date = pd.to_datetime(spot_date)
    
    mortgages_df['Datum Ingang Leningdeel'] = pd.to_datetime(mortgages_df['Datum Ingang Leningdeel'])
    mortgages_df['Mortgage_age'] = (spot_date - mortgages_df['Datum Ingang Leningdeel']).dt.days / 365.25
    
    n_months = 360
    aging_matrices = {}

    for i, row in mortgages_df.iterrows():
        start_age = row['Mortgage_age']
        rvp_end_date = pd.to_datetime(row['Datum Eind RVP'])
        months_active = min((rvp_end_date.year - spot_date.year) * 12 + rvp_end_date.month - spot_date.month, n_months)

        aging_matrix = np.zeros(n_months)
        exact_age_years = (spot_date - row['Datum Ingang Leningdeel']).days / 365.25

        for month in range(months_active):
            aging_matrix[month] = exact_age_years + month / 12.0

        aging_matrices[f'Hypotheek {i + 1}'] = aging_matrix

    return mortgages_df, aging_matrices


# In[7]:


spot_date = '2023-03-31'
mortgages_df, aging_matrices = calculate_aging(mortgages_df, spot_date)

# Toon de resultaten voor de eerste paar hypotheken
for i in range(len(mortgages_df)):
    print(f"Leeftijdsmatrix voor hypotheek {i + 1}:")
    print(aging_matrices[f'Hypotheek {i + 1}'])


# In[8]:


def determine_segment(row):
    if 'ProductLijn' in row and 'Overbrugging' in row['ProductLijn']:
        row['PrepaymentType'] = 'bridge loans'
    if 'Merk' in row and row['Merk'] in ['Staal euro lening of RC', 'Acier']:
        row['PrepaymentType'] = 'Acier'

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
        return 'Segment 9'

if 'Fixed' not in mortgages_df.columns:
    mortgages_df['Fixed'] = 0

mortgages_df['Segment'] = mortgages_df.apply(determine_segment, axis=1)


# In[9]:


age_parameters = incentives_df[incentives_df['Coefficient'].isin(['e_age', 'f_age'])]
age_parameters = age_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]
age_parameters = age_parameters.set_index('Coefficient').apply(pd.to_numeric)

def calculate_age_factor(age, e_age, f_age):
    return min(1, e_age + f_age * age)

age_factor_matrices = {}

for i, row in mortgages_df.iterrows():
    segment = row['Segment']
    e_age = age_parameters.loc['e_age', segment]
    f_age = age_parameters.loc['f_age', segment]
    aging_matrix = aging_matrices[f'Hypotheek {i + 1}']
    age_factor_matrix = np.zeros_like(aging_matrix)
    
    for month in range(len(aging_matrix)):
        age_factor_matrix[month] = calculate_age_factor(aging_matrix[month], e_age, f_age)
    
    rvp_end_date = pd.to_datetime(row['Datum Eind RVP'])
    rvp_index = min(int((rvp_end_date - pd.to_datetime(spot_date)).days / 30.44) + 1, len(aging_matrix))
    
    if rvp_index < len(aging_matrix):
        age_factor_matrix[rvp_index:] = 0
    
    age_factor_matrices[f'Hypotheek {i + 1}'] = age_factor_matrix


# In[10]:


# Toon de age factor matrices voor de eerste paar hypotheken
for i in range(len(mortgages_df)):
    print(f"Age factor matrix voor hypotheek {i + 1}:")
    print(age_factor_matrices[f'Hypotheek {i + 1}'])

# In[25]:
print(mortgages_df[['Aflosvorm', 'LTV Unindexed', 'Datum Ingang Leningdeel', 'Mortgage_age', 'Segment']])


# In[11]:


seasonality_parameters = incentives_df[incentives_df['Coefficient'].str.startswith('season_')]
seasonality_parameters = seasonality_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]

# In[27]:
def create_seasonality_matrix(mortgages_df, spot_date, seasonality_parameters):
    spot_date = pd.to_datetime(spot_date)
    n_months = 360
    seasonality_matrices = {}

    for i, row in mortgages_df.iterrows():
        segment = row['Segment']
        segment_index = segment.replace('Segment ', '')
        segment_col = f'Segment {segment_index}'
        
        season_factors = pd.to_numeric(seasonality_parameters[segment_col].values)
        season_factors = season_factors + 1
        
        rvp_end_date = pd.to_datetime(row['Datum Eind RVP'])
        months_active = min((rvp_end_date.year - spot_date.year) * 12 + rvp_end_date.month - spot_date.month, n_months)

        seasonality_matrix = np.zeros(n_months)
        start_month = (spot_date.month % 12)
        
        for month in range(1, months_active + 1):
            month_index = (start_month + month - 1) % 12
            seasonality_matrix[month - 1] = season_factors[month_index]

        seasonality_matrices[f'Hypotheek {i + 1}'] = seasonality_matrix

    return seasonality_matrices


# In[12]:


seasonality_matrices = create_seasonality_matrix(mortgages_df, spot_date, seasonality_parameters)

# Print de seasonality matrix voor elke hypotheek
for hypotheek, matrix in seasonality_matrices.items():
    print(f"{hypotheek}:")
    print(matrix)


# In[13]:


rente_nominaal = mortgages_df['Rente Nominaal']
print(rente_nominaal.head(25))

# In[29]:
spotdate = pd.to_datetime('2023-03-31')
mortgages_df['Datum Eind RVP'] = pd.to_datetime(mortgages_df['Datum Eind RVP'])
mortgages_df['RENTEVASTPERIODE'] = np.ceil((mortgages_df['Datum Eind RVP'] - spotdate) / np.timedelta64(1, 'Y')).astype(int).astype(str) + ' jaar'
print(mortgages_df[['Datum Ingang Leningdeel', 'Datum Eind RVP', 'RENTEVASTPERIODE']].head(20))

# In[31]:
mortgages_df['Datum Eind RVP'] = pd.to_datetime(mortgages_df['Datum Eind RVP'], errors='coerce')
mortgages_df['initial_remainingRVP'] = np.ceil((mortgages_df['Datum Eind RVP'] - spotdate) / np.timedelta64(1, 'M')).astype(int)

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


# In[14]:


aflosvorm_mapping = {
    'Annuiteit': 'Annuïtair',
    'Lineair': 'Lineair',
    'Aflossingsvrij': 'Aflossingsvrij',
    'Levensverzekering': 'Aflossingsvrij',
    'Spaarrekening': 'Aflossingsvrij'
}

def normalize_aflosvorm(aflosvorm):
    return aflosvorm_mapping.get(aflosvorm, 'Aflossingsvrij')


# In[15]:


def round_nearest_rvp(rvp):
    # Haal unieke rentevastperiodes uit de rates_df en sorteer ze als integers
    rvp_categories = rates_df['RENTEVASTPERIODE'].unique()
    rvp_categories = sorted([int(cat.split()[0]) for cat in rvp_categories])
    
    # Bereken het verschil tussen de gegeven rvp en elke categorie
    nearest_category = min(rvp_categories, key=lambda x: abs(rvp - x))
    return f"{nearest_category} jaar"


def get_average_rate(merk, aflosvorm, rvp, risk_class):
    relevant_rates = rates_df[(rates_df['namabd'] == merk) &
                              (rates_df['productvorm'] == aflosvorm) &
                              (rates_df['RENTEVASTPERIODE'] == rvp) &
                              (rates_df['RISICOKLASSE'].str.strip() == risk_class)]
    
    if not relevant_rates.empty:
        return relevant_rates.iloc[0]['Average of RENTE'] * 0.01
    
    # Gebruik de aangepaste afrondingsmethode naar de dichtstbijzijnde categorie
    rounded_rvp = round_nearest_rvp(int(rvp.split()[0]))
    relevant_rates = rates_df[(rates_df['namabd'] == merk) &
                              (rates_df['productvorm'] == aflosvorm) &
                              (rates_df['RENTEVASTPERIODE'] == rounded_rvp) &
                              (rates_df['RISICOKLASSE'].str.strip() == risk_class)]
    
    if not relevant_rates.empty:
        return relevant_rates.iloc[0]['Average of RENTE'] * 0.01
    
    return np.nan


# In[16]:


def create_interest_rate_matrix(mortgages_df, months=360):
    n_hypotheken = len(mortgages_df)
    rate_matrix = np.zeros((n_hypotheken, months))
    
    for i, row in mortgages_df.iterrows():
        risk_class = determine_risk_class(row['LTV Unindexed'])
        aflosvorm = normalize_aflosvorm(row['Aflosvorm'])
        remaining_rvp = row['initial_remainingRVP']
        
        for month in range(months):
            # Bepaal de overgebleven rvp in jaren
            if month < remaining_rvp:
                current_rvp = (remaining_rvp - month - 1) / 12  # Pas de grens iets aan om afronding vroeger te laten gebeuren
                rvp_category = round_nearest_rvp(current_rvp)
                rate = get_average_rate(row['Merk'], aflosvorm, rvp_category, risk_class)
                rate_matrix[i, month] = rate if rate is not np.nan else 0
            else:
                rate_matrix[i, month] = 0
    
    return rate_matrix





# In[17]:


interest_rate_matrix = create_interest_rate_matrix(mortgages_df)

for idx, matrix in enumerate(interest_rate_matrix):
    print(f"Hypotheek {idx + 1}:")
    print(matrix)
    print("\n")


# In[18]:


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

nominal_rate_matrix = create_nominal_rate_matrix(mortgages_df)
incentive_matrix = nominal_rate_matrix - interest_rate_matrix

for idx, matrix in enumerate(incentive_matrix):
    print(f"Incentive Hypotheek {idx + 1}:")
    print(matrix)
    print("\n")


# In[19]:


incentive_parameters = incentives_df[incentives_df['Coefficient'].isin(['a_inc', 'b_inc', 'c_inc', 'd_inc'])]
incentive_parameters = incentive_parameters[['Coefficient', 'Segment 1a', 'Segment 1b', 'Segment 2', 'Segment 3', 'Segment 4', 'Segment 5', 'Segment 6', 'Segment 7', 'Segment 8', 'Segment 9']]
incentive_parameters = incentive_parameters.set_index('Coefficient').apply(pd.to_numeric)
def calculate_incentive_factor(incentive, a_inc, b_inc, c_inc, d_inc):
    # Bereken de incentive-factor volgens de MATLAB-formule
    return a_inc + b_inc / (1 + np.exp(c_inc + d_inc * incentive))

# Genereer de incentive factor matrices
incentive_factor_matrices = {}

for i, row in mortgages_df.iterrows():
    segment = row['Segment']
    # Haal de juiste parameterwaarden op voor het huidige segment
    a_inc = incentive_parameters.loc['a_inc', segment]
    b_inc = incentive_parameters.loc['b_inc', segment]
    c_inc = incentive_parameters.loc['c_inc', segment]
    d_inc = incentive_parameters.loc['d_inc', segment]
    
    # Incentive matrix voor de specifieke hypotheek
    incentive_matrix_for_loan = incentive_matrix[i]
    incentive_factor_matrix = np.zeros_like(incentive_matrix_for_loan)
    
    # Bereken de incentive factor voor elke maand
    for month in range(len(incentive_matrix_for_loan)):
        incentive_factor_matrix[month] = calculate_incentive_factor(
            incentive_matrix_for_loan[month], a_inc, b_inc, c_inc, d_inc
        )
    
    # Sla de berekende incentive factor matrix op
    incentive_factor_matrices[f'Hypotheek {i + 1}'] = incentive_factor_matrix

# Print de Incentive Factor matrix voor controle
for i in range(len(mortgages_df)):
    print(f"Incentive Factor Matrix voor Hypotheek {i + 1}:")
    print(incentive_factor_matrices[f'Hypotheek {i + 1}'])


# In[20]:


def create_smm_matrix(age_matrices, seasonality_matrices, incentive_matrices, mortgages_df):
    smm_matrices = {}
    
    for i in range(len(mortgages_df)):
        hypotheek_key = f'Hypotheek {i + 1}'
        age_matrix = age_matrices[hypotheek_key]
        seasonality_matrix = seasonality_matrices[hypotheek_key]
        incentive_matrix = incentive_matrices[hypotheek_key]
        
        assert age_matrix.shape == seasonality_matrix.shape == incentive_matrix.shape, "Matrices have different shapes"
        
        smm_matrix = age_matrix * seasonality_matrix * incentive_matrix
        smm_matrices[hypotheek_key] = smm_matrix
    
    return smm_matrices


# In[21]:


def calculate_survival_matrix(smm_matrices):
    survival_matrices = {}
    
    for hypotheek, smm_matrix in smm_matrices.items():
        survival_matrix = np.cumprod(1 - smm_matrix, axis=0)
        survival_matrices[hypotheek] = survival_matrix
    
    return survival_matrices


# In[22]:


smm_matrices = create_smm_matrix(age_factor_matrices, seasonality_matrices, incentive_factor_matrices, mortgages_df)
survival_matrices = calculate_survival_matrix(smm_matrices)

# Print de SMM-matrix en survival-matrix voor elke hypotheek
for hypotheek, matrix in smm_matrices.items():
    print(f"SMM Matrix voor {hypotheek}:")
    print(matrix)
    print("\n")

for hypotheek, matrix in survival_matrices.items():
    print(f"Survival Matrix voor {hypotheek}:")
    print(matrix)
    print("\n")


# In[23]:


with open('smm_matrices.pkl', 'wb') as f:
    pickle.dump(smm_matrices, f)

with open('survival_matrices.pkl', 'wb') as f:
    pickle.dump(survival_matrices, f)


# In[ ]:




