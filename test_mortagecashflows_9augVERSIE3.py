#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Bestanden inlezen
mortgages_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Test_mortgages.csv'

# Lees de CSV-bestanden in
mortgages_df = pd.read_csv(mortgages_path, delimiter=',', encoding='ISO-8859-1')

# Datum parsing functie
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
mortgages_df['Datum Eind RVP'] = mortgages_df['Datum Eind RVP'].apply(parse_dates)
mortgages_df['Datum Eind Leningdeel'] = mortgages_df['Datum Eind Leningdeel'].apply(parse_dates)

# Spotdate instellen
spotdate = pd.Timestamp('2023-03-31')

print(mortgages_df)


# In[2]:


unique_rvps = mortgages_df['Looptijd Rentevastperiode 1'].unique()
unique_aflosvorm = mortgages_df['Aflosvorm'].unique()
print("Unieke Rentevaste Periodes:", unique_rvps)
print("Unieke Aflosvormen:", unique_aflosvorm)




# In[3]:


num_steps = 360
num_morts = len(mortgages_df)

# Bepaal de resterende maanden vanaf de spotdate
mortgages_df['Months After Spotdate'] = (mortgages_df['Datum Eind RVP'].dt.to_period('M') - spotdate.to_period('M')).apply(lambda x: max(0, x.n))

# Pas n_periods aan naar de resterende maanden vanaf de spotdate
mortgages_df['n_periods_spotdate'] = mortgages_df[['Months After Spotdate', 'Looptijd Rentevastperiode 1']].min(axis=1)

# Gebruik de resterende hoofdsom vanaf de spotdate
mortgages_df['Initial Principal'] = mortgages_df['Hoofdsom Restant']

print("Aantal hypotheken:", num_morts)

issue_dt = pd.to_datetime(mortgages_df['Datum Ingang Leningdeel'])
mat_dt = pd.to_datetime(mortgages_df['Datum Eind Leningdeel'])
reprice_dt = pd.to_datetime(mortgages_df['Datum Eind RVP'])
fixed_per = mortgages_df['Looptijd Rentevastperiode 1']
orig_principal = mortgages_df['Hoofdsom Oorspronkelijk']
curr_principal = mortgages_df['Hoofdsom Restant']
int_rate = mortgages_df['Rente Nominaal']
red_types = mortgages_df['Aflosvorm']
remaining_rvp = mortgages_df['Looptijd Rentevastperiode 1']
principal_savings = mortgages_df['Bedrag Banksparen'] + mortgages_df['Bedrag Spaardepot']

# Initieer arrays voor de resultaten
tot_principal = np.zeros((num_morts, num_steps))
tot_redemptions = np.zeros((num_morts, num_steps))
tot_resets = np.zeros((num_morts, num_steps))
tot_interest = np.zeros((num_morts, num_steps))
tot_count = np.zeros((num_morts, num_steps))
mort_types = [''] * num_morts
count_mort_types = np.zeros(num_morts)
tot_fltr = np.zeros(len(mortgages_df), dtype=bool)
now_dt = pd.Timestamp('2023-03-31').toordinal()

# Controleer variabelen en initialisaties
print("Aantal hypotheken:", num_morts)


# In[4]:


# Consistente mapping tussen beide scripts
aflosvorm_mapping = {
    'Annuiteit': 'annuity',
    'Lineair': 'linear',
    'Aflossingsvrij': 'bullet',
    'Levensverzekering': 'bullet',  # Behandeld als aflossingsvrij
    'Spaarrekening': 'bullet',  # Behandeld als aflossingsvrij
    'Spaarverzekering': 'savings',
    'Beleggingshypotheek': 'investment',
    'Belegging': 'investment',
    'Leven': 'savings',
    'Beleggingsverzekering': 'investment'
}
print("Aflosvorm Mapping:", aflosvorm_mapping)


# In[5]:


def annuity_cashflows(pv, rate, n_periods):
    interest = np.zeros(n_periods)
    principal = np.zeros(n_periods)
    balance = np.zeros(n_periods)
    annuity_payment = pv * (rate / (1 - (1 + rate) ** -n_periods))
    remaining_balance = pv

    for t in range(n_periods):
        interest[t] = remaining_balance * rate
        principal[t] = annuity_payment - interest[t]
        remaining_balance -= principal[t]
        balance[t] = remaining_balance
    
    # Zorg ervoor dat de laatste betaling de resterende balans volledig aflost
    if n_periods > 0:
        principal[-1] += balance[-2]
        balance[-1] = 0

    return interest, principal, balance


# In[6]:


def linear_cashflows(pv, rate, n_periods):
    interest = np.zeros(n_periods)
    principal = np.full(n_periods, pv / n_periods)
    balance = np.zeros(n_periods)
    remaining_balance = pv

    for t in range(n_periods):
        interest[t] = remaining_balance * rate
        remaining_balance -= principal[t]
        balance[t] = remaining_balance
    
    # Zorg ervoor dat de laatste betaling de resterende balans volledig aflost
    if n_periods > 0:
        principal[-1] += balance[-2]
        balance[-1] = 0

    return interest, principal, balance



# In[7]:


def interest_only_cashflows(pv, rate, n_periods):
    interest = np.full(n_periods, pv * rate)
    principal = np.zeros(n_periods)
    balance = np.full(n_periods, pv)
    
    # Voeg de volledige hoofdsom toe aan de laatste maand
    if n_periods > 0:
        principal[-1] = pv  # De laatste maand wordt de volledige aflossing van het restant
        balance[-1] = 0  # Na de laatste betaling is het saldo 0

    return interest, principal, balance




# In[8]:


def savings_cashflows(pv, rate, n_periods):
    interest = np.zeros(n_periods)
    principal = np.zeros(n_periods)
    balance = np.zeros(n_periods)
    # Logic for savings cashflows can be added here
    return interest, principal, balance

def investment_cashflows(pv, rate, n_periods):
    interest = np.zeros(n_periods)
    principal = np.zeros(n_periods)
    balance = np.zeros(n_periods)
    # Logic for investment cashflows can be added here
    return interest, principal, balance


# In[9]:


def calculate_cashflows(mortgages_df, spotdate):
    cashflows = []
    num_steps = 360

    for idx, row in mortgages_df.iterrows():
        aflosvorm = aflosvorm_mapping.get(row['Aflosvorm'], 'unknown')
        n_periods = row['n_periods_spotdate']
        
        if aflosvorm == 'annuity':
            interest, principal, balance = annuity_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods)
        elif aflosvorm == 'linear':
            interest, principal, balance = linear_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods)
        elif aflosvorm == 'bullet' or aflosvorm == 'Aflossingsvrij':
            interest, principal, balance = interest_only_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods)
        elif aflosvorm == 'savings' or aflosvorm == 'Levensverzekering':
            interest, principal, balance = savings_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods)
        elif aflosvorm == 'investment':
            interest, principal, balance = investment_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods)
        else:
            raise ValueError(f"Onbekende aflosvorm: {row['Aflosvorm']}")

        # Vul de resterende periodes met nullen (om te zorgen dat alle arrays 360 lang zijn)
        interest = np.pad(interest, (0, num_steps - len(interest)), 'constant')
        principal = np.pad(principal, (0, num_steps - len(principal)), 'constant')
        balance = np.pad(balance, (0, num_steps - len(balance)), 'constant')

        cashflows.append({
            'HypotheekID': row['Leningdeel'],
            'Aflosvorm': aflosvorm,
            'Interest': interest,
            'Principal': principal,
            'Balance': balance
        })

    return cashflows




# In[10]:


# Cashflows berekenen
cashflows = calculate_cashflows(mortgages_df, spotdate)

# Print de cashflows voor alle hypotheken
for idx, cashflow in enumerate(cashflows):
    print(f"Hypotheek {idx + 1}:")
    print(f"Hypotheek ID: {cashflow['HypotheekID']}")
    print(f"Aflosvorm: {cashflow['Aflosvorm']}")
    print("Interest Cashflows:", cashflow['Interest'])
    print("Principal Cashflows:", cashflow['Principal'])
    print("Balance:", cashflow['Balance'])
    print("\n")



# In[ ]:





# In[ ]:




