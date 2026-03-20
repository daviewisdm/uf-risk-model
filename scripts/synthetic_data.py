import numpy as np
import pandas as pd

# Define number of samples (adjust as needed)
n_samples = 2000

# Set seed for reproducibility
np.random.seed(42)

# Generate features based on realistic distributions
age = np.random.randint(15, 50, n_samples)

# Race Distribution
race = np.random.choice(
    ['Black', 'Indian', 'White', 'Hispanic', 'Asian', 'Other'],
    n_samples,
    p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.05]
)

bmi = np.random.normal(27, 5, n_samples).clip(15, 50)
parity = np.random.poisson(1.5, n_samples).clip(0, 5)
menarche_age = np.random.randint(10, 16, n_samples)
hypertension = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
pcos = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
vitamin_d_deficient = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
diet_quality = np.random.uniform(0, 10, n_samples)
physical_activity = np.random.exponential(5, n_samples).clip(0, 20)
smoking = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
family_history = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
stress_level = np.random.randint(1, 11, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Race': race,
    'BMI': bmi,
    'Parity': parity,
    'Menarche_Age': menarche_age,
    'Hypertension': hypertension,
    'PCOS': pcos,
    'Vitamin_D_Deficient': vitamin_d_deficient,
    'Diet_Quality': diet_quality,
    'Physical_Activity': physical_activity,
    'Smoking': smoking,
    'Family_History': family_history,
    'Stress_Level': stress_level
})

# Calculate probability of fibroids based on risk factors (aligned with epidemiology)
df['prob'] = 0.3  # Base prevalence
df['prob'] += (df['Race'] == 'Black') * 0.30        # Highest risk
df['prob'] += (df['Race'] == 'Indian') * 0.20       # Second highest risk
df['prob'] += (df['Race'].isin(['Asian', 'Hispanic'])) * 0.10
df['prob'] += (df['Age'] > 35) * 0.10
df['prob'] += (df['BMI'] > 30) * 0.15
df['prob'] += (df['Parity'] == 0) * 0.10
df['prob'] += (df['Menarche_Age'] < 12) * 0.05
df['prob'] += df['Hypertension'] * 0.10
df['prob'] += df['PCOS'] * 0.15
df['prob'] += df['Vitamin_D_Deficient'] * 0.10
df['prob'] += (df['Diet_Quality'] < 5) * 0.05
df['prob'] += (df['Physical_Activity'] < 5) * 0.05
df['prob'] += (df['Smoking'] == 0) * 0.05           # Simplified protective effect
df['prob'] += df['Family_History'] * 0.20
df['prob'] += (df['Stress_Level'] > 7) * 0.05

# Clip to [0,1] and generate binary target
df['prob'] = df['prob'].clip(0, 1)
df['Has_Fibroids'] = np.random.binomial(1, df['prob'])

# Drop temporary prob column
df = df.drop('prob', axis=1)

# Save to CSV
df.to_csv('synthetic_fibroids_data.csv', index=False)

# Preview
print(df.head())
print("\nRace distribution:")
print(df['Race'].value_counts(normalize=True).round(3))
print("\nFibroid prevalence:")
print(df['Has_Fibroids'].value_counts(normalize=True))