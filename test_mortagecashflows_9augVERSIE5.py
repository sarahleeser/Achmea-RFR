#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

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


# Laad de survival-matrices in je hoofdscript
with open('survival_matrices.pkl', 'rb') as f:
    survival_matrices = pickle.load(f)


# In[3]:


# Unieke waardes bekijken
unique_rvps = mortgages_df['Looptijd Rentevastperiode 1'].unique()
unique_aflosvorm = mortgages_df['Aflosvorm'].unique()
print("Unieke Rentevaste Periodes:", unique_rvps)
print("Unieke Aflosvormen:", unique_aflosvorm)


# In[4]:


num_steps = 360
num_morts = len(mortgages_df)

# Bepaal de resterende maanden vanaf de spotdate
mortgages_df['Months After Spotdate'] = (mortgages_df['Datum Eind RVP'].dt.to_period('M') - spotdate.to_period('M')).apply(lambda x: max(0, x.n))

# Pas n_periods aan naar de resterende maanden vanaf de spotdate
mortgages_df['n_periods_spotdate'] = mortgages_df[['Months After Spotdate', 'Looptijd Rentevastperiode 1']].min(axis=1)

# Bepaal de totale looptijd van de lening in maanden door het aantal dagen te delen door 30
mortgages_df['n_periods_total'] = (mortgages_df['Datum Eind Leningdeel'] - mortgages_df['Datum Ingang Leningdeel']).dt.days // 30

# Gebruik de resterende hoofdsom vanaf de spotdate
mortgages_df['Initial Principal'] = mortgages_df['Hoofdsom Restant']

print("Aantal hypotheken:", num_morts)


# In[5]:


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


# In[6]:


def annuity_cashflows(pv, rate, n_periods_rvp, n_periods_total, months_after_spotdate):
    # Bereken de maandelijkse annuïteitenbetaling over de resterende looptijd
    n_remaining_periods = n_periods_total - months_after_spotdate
    annuity_payment = pv * (rate / (1 - (1 + rate) ** -n_remaining_periods))

    # Arrays voor interest, principal en balance
    interest = np.zeros(n_periods_rvp)
    principal = np.zeros(n_periods_rvp)
    balance = np.zeros(n_periods_rvp)
    
    # Begin met het oorspronkelijke bedrag
    remaining_balance = pv

    for t in range(n_periods_rvp):
        # Bereken de interest op het resterende saldo
        interest[t] = remaining_balance * rate
        # Bereken de aflossing als deel van de annuïteitenbetaling
        principal[t] = annuity_payment - interest[t]
        # Werk het resterende saldo bij
        remaining_balance -= principal[t]
        balance[t] = remaining_balance

    # Zorg ervoor dat de laatste betaling de resterende balans volledig aflost
    if remaining_balance > 0:
        principal[-1] += remaining_balance
        balance[-1] = 0

    return interest, principal, balance






# In[7]:


def linear_cashflows(pv, rate, n_periods_rvp, n_periods_total, months_after_spotdate):
    # Bereken de maandelijkse aflossing over de resterende looptijd
    n_remaining_periods = n_periods_total - months_after_spotdate
    monthly_principal_payment = pv / n_remaining_periods

    # Arrays voor interest, principal en balance
    interest = np.zeros(n_periods_rvp)
    principal = np.zeros(n_periods_rvp)
    balance = np.zeros(n_periods_rvp)
    
    # Begin met het oorspronkelijke bedrag
    remaining_balance = pv

    for t in range(n_periods_rvp):
        # Bereken de interest op het resterende saldo
        interest[t] = remaining_balance * rate
        # Maandelijkse aflossing is gelijk voor alle maanden
        principal[t] = monthly_principal_payment
        # Werk het resterende saldo bij
        remaining_balance -= principal[t]
        balance[t] = remaining_balance

    # Zorg ervoor dat de laatste betaling de resterende balans volledig aflost
    if remaining_balance > 0:
        principal[-1] += remaining_balance
        balance[-1] = 0

    return interest, principal, balance


# In[8]:


def interest_only_cashflows(pv, rate, n_periods_rvp, n_periods_total):
    interest = np.full(n_periods_rvp, pv * rate)
    principal = np.zeros(n_periods_rvp)
    balance = np.full(n_periods_rvp, pv)
    
    # Voeg de volledige hoofdsom toe aan de laatste maand
    if n_periods_rvp > 0:
        principal[-1] = pv  # De laatste maand wordt de volledige aflossing van het restant
        balance[-1] = 0  # Na de laatste betaling is het saldo 0

    return interest, principal, balance


# In[9]:


def savings_cashflows(pv, rate, n_periods_rvp, n_periods_total):
    interest = np.zeros(n_periods_rvp)
    principal = np.zeros(n_periods_rvp)
    balance = np.zeros(n_periods_rvp)
    # Logic for savings cashflows can be added here
    return interest, principal, balance

def investment_cashflows(pv, rate, n_periods_rvp, n_periods_total):
    interest = np.zeros(n_periods_rvp)
    principal = np.zeros(n_periods_rvp)
    balance = np.zeros(n_periods_rvp)
    # Logic for investment cashflows can be added here
    return interest, principal, balance


# In[11]:


def calculate_cashflows_with_survival(mortgages_df, spotdate, survival_matrices):
    cashflows = []
    num_steps = 360

    for idx, row in mortgages_df.iterrows():
        aflosvorm = aflosvorm_mapping.get(row['Aflosvorm'], 'unknown')
        n_periods_rvp = row['n_periods_spotdate']
        n_periods_total = row['n_periods_total']
        months_after_spotdate = (spotdate.to_period('M') - row['Datum Ingang Leningdeel'].to_period('M')).n
        survival_matrix = survival_matrices[f'Hypotheek {idx}']

        if aflosvorm == 'annuity':
            interest, principal, balance = annuity_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, n_periods_total, months_after_spotdate)
        elif aflosvorm == 'linear':
            interest, principal, balance = linear_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, n_periods_total, months_after_spotdate)
        elif aflosvorm == 'bullet' or aflosvorm == 'Aflossingsvrij':
            interest, principal, balance = interest_only_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, n_periods_total)
        elif aflosvorm == 'savings' or aflosvorm == 'Levensverzekering':
            interest, principal, balance = savings_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, n_periods_total)
        elif aflosvorm == 'investment':
            interest, principal, balance = investment_cashflows(row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, n_periods_total)
        else:
            raise ValueError(f"Onbekende aflosvorm: {row['Aflosvorm']}")

        # Pas de cashflows aan met de survival-matrix
        interest = interest * survival_matrix[:len(interest)]
        principal = principal * survival_matrix[:len(principal)]
        balance = balance * survival_matrix[:len(balance)]

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


# In[12]:


# Bereken de cashflows met survival
cashflows_with_survival = calculate_cashflows_with_survival(mortgages_df, spotdate, survival_matrices)

# Print de cashflows voor alle hypotheken
for idx, cashflow in enumerate(cashflows_with_survival):
    print(f"Hypotheek {idx + 1}:")
    print(f"Hypotheek ID: {cashflow['HypotheekID']}")
    print(f"Aflosvorm: {cashflow['Aflosvorm']}")
    print("Interest Cashflows:", cashflow['Interest'])
    print("Principal Cashflows:", cashflow['Principal'])
    print("Balance:", cashflow['Balance'])
    print("\n")


# In[ ]:





# In[ ]:




