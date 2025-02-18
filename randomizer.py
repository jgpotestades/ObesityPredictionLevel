import pandas as pd
import numpy as np
import random

def generate_random_data(num_rows, output_filename="random_obesity_data.csv"):
    """Generates random data for obesity prediction and saves it to a CSV file.
       Heights are generated in centimeters.

    Args:
        num_rows (int): The number of rows to generate.
        output_filename (str): The name of the CSV file to save.
    """

    # Define possible values for each column (mimicking your provided data)
    sex = ['Female', 'Male', 'Other']
    heights_cm = np.random.uniform(150, 200, num_rows)  # Realistic height range (cm)
    weights = np.random.randint(50, 120, num_rows)  # Realistic weight range (kg)
    calcs = ['No', 'Sometimes', 'Frequently', 'Always']
    favcs = ['No', 'Sometimes', 'Frequently']
    fcvcs = np.random.randint(1, 4, num_rows)  # Numerical values 1-3
    ncps = np.random.randint(1, 6, num_rows)  # Numerical values 1-5
    sccs = ['No', 'Yes']
    smokes = ['No', 'Yes']
    ch2os = np.random.uniform(1, 4, num_rows)  # Numerical values 1-3
    family_history = ['No', 'Yes']
    fafs = np.random.uniform(0, 4, num_rows)  # Realistic FAF range
    tues = np.random.uniform(0, 3, num_rows)  # Realistic TUE range
    caecs = ['No', 'Sometimes', 'Frequently', 'Always']
    mtrans = ['Automobile', 'Bike', 'Public Transportation', 'Walking', 'Motorbike', 'Other']
    classif = ['Normal Weight', 'Overweight Level I', 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III']

    # Create a dictionary to store the data
    data = {
        'Age': np.random.randint(18, 65, num_rows),  # Realistic age range
        'Sex': random.choices(sex, k=num_rows),
        'Height': heights_cm,  # Heights in cm
        'Weight': weights,
        'CALC': random.choices(calcs, k=num_rows),
        'FAVC': random.choices(favcs, k=num_rows),
        'FCVC': fcvcs,
        'NCP': ncps,
        'SCC': random.choices(sccs, k=num_rows),
        'SMOKE': random.choices(smokes, k=num_rows),
        'CH2O': ch2os,
        'FAMHIS': random.choices(family_history, k=num_rows),
        'FAF': fafs,
        'TUE': tues,
        'CAEC': random.choices(caecs, k=num_rows),
        'MTRANS': random.choices(mtrans, k=num_rows),
        'Classif': random.choices(classif, k=num_rows)
    }

    # Create a Pandas DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"Random data saved to {output_filename}")


# Example usage:
num_rows_to_generate = 1000  # You can adjust this
generate_random_data(num_rows_to_generate)  # Saves to random_obesity_data.csv by default

# To save with a different name:
# generate_random_data(num_rows_to_generate, "my_random_data.csv")