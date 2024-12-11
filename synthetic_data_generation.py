import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

n_rows = 10000

# gt graph:

#  A->C, B->C, C->D, C->E
def gen_single_relation(n_rows=10000):
    # Generate independent variables A and B (normally distributed)
    A = np.random.normal(loc=0, scale=1, size=n_rows)  # Mean 0, Std 1
    B = np.random.normal(loc=0, scale=1, size=n_rows)  # Mean 0, Std 1

    # Generate dependent variable C (dependent on A and B)
    # C = 0.5*A + 0.3*B + some noise
    C = 0.5 * A + 0.3 * B + np.random.normal(loc=0, scale=0.1, size=n_rows)

    # Generate dependent variable D (dependent on C)
    # D = 0.7*C + some noise
    D = 0.7 * C + np.random.normal(loc=0, scale=0.1, size=n_rows)

    # Generate dependent variable E (dependent on C)
    # E = -0.4*C + some noise
    E = -0.4 * C + np.random.normal(loc=0, scale=0.1, size=n_rows)

    data = pd.DataFrame({
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'E': E
    })

    # data.to_csv('synthetic_data.csv', index=False)

    return data


def gen_single_relation_with_hidden_confounder(l_to_a=0.6, l_to_b=0.4, n_rows=10000):

    # Generate the latent confounder L (normally distributed)
    L = np.random.normal(loc=0, scale=1, size=n_rows)  # Mean 0, Std 1

    # Generate A and B influenced by the latent confounder L
    A = l_to_a * L + np.random.normal(loc=0, scale=1, size=n_rows)  # L heavily influences A
    B = l_to_b * L + np.random.normal(loc=0, scale=1, size=n_rows)  # L moderately influences B

    # Generate C influenced by A and B, but not by L
    C = A + B + np.random.normal(loc=0, scale=0.1, size=n_rows)

    # Generate D influenced by C
    D = 0.7 * C + np.random.normal(loc=0, scale=0.1, size=n_rows)

    # Generate E influenced by C
    E = -0.4 * C + np.random.normal(loc=0, scale=0.1, size=n_rows)

    # Combine into a pandas DataFrame, omitting the latent variable L
    data = pd.DataFrame({
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'E': E
    })

    return data