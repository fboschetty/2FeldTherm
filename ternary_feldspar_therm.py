import numpy as np
import pandas as pd
from scipy import optimize
import itertools

# Column and af_row names
cols = ["WH", "WS", "WV"]
rows = ["WAbOr", "WOrAb", "WAbAn", "WAnAb", "WOrAn", "WAnOr", "WAbOrAn"]

# Nekvasil and Burnham 1987
NB1987 = pd.DataFrame([
    [30978.0,  21.40,  0.361],
    [17062.0,   0.0,   0.361],
    [14129.4,   6.14,  0.0],
    [11225.7,   7.87,  0.0],
    [25030.3, -10.80,  0.0],
    [75023.3,  22.97,  0.0],
    [    0.0,   0.0,   0.0]
], columns=cols, index=rows)

# Lindsley and Nekvasil 1988
LN1988 = pd.DataFrame([
    [18810, 10.30,   0.4602],
    [27320, 10.30,   0.3264],
    [14129,  6.18,   0.0],
    [11226,  7.87,   0.0],
    [33415,  0.0,    0.0],
    [43369,  8.43, -0.1037],
    [19969,  0.0,  -1.0950]
], columns=cols, index=rows)

# Ghiorso 1984
G1984 = pd.DataFrame([
    [ 30978,  21.40, 0.361],
    [ 17062,   0.0,  0.361],
    [ 28226,   0.0,  0.0],
    [  8741,   0.0,  0.0],
    [ 67469, -20.21, 0.0],
    [ 27983,  11.06, 0.0],
    [-13869, -14.63, 0.0]
], columns=cols, index=rows)

# Green and Usdansky 1986
GU1986 = pd.DataFrame([
    [ 18810,   10.3,    0.364],
    [ 27320,   10.3,    0.364],
    [ 28226,    0.0,    0.0],
    [  8743,    0.0,    0.0],
    [ 65305, -114.104,  0.9699],
    [-65407,   12.537,  2.1121],
    [   0.0,    0.0,   -1.094]
], columns=cols, index=rows)

# Fuhrman and Lindsley 1988
FL1988 = pd.DataFrame([
    [18810, 10.3,  0.394],
    [27320, 10.3,  0.394],
    [28226,  0.0,  0.0],
    [ 8741,  0.0,  0.0],
    [52468,  0.0,  0.0],
    [47396,  0.0, -0.120],
    [ 8700,  0.0, -1.094]
], columns=cols, index=rows)

# Elkins and Grove 1990
EG1990 = pd.DataFrame([
    [18810, 10.3,  0.4602],
    [27320, 10.3,  0.3264],
    [ 7924,  0.0,  0.0],
    [  0.0,  0.0,  0.0],
    [40317,  0.0,  0.0],
    [38974,  0.0, -0.1037],
    [12545,  0.0, -1.095]
], columns=cols, index=rows)

# Benisek et al 2004 Aluminium-Avoidance
B2004_Al = pd.DataFrame([
    [19550, 10.5,  0.327],
    [22820,  6.3,  0.461],
    [31000,  4.5,  0.069],
    [ 9800, -1.7, -0.049],
    [90600, 29.5, -0.257],
    [60300, 11.2, -0.210],
    [ 8000,  0.0, -0.467]
], columns=cols, index=rows)

# Benisek et al 2004 Molecular Mixing
B2004_MM = pd.DataFrame([
    [19550, 10.5,  0.327],
    [22820,  6.3,  0.461],
    [31000, 19.0,  0.069],
    [ 9800,  7.5, -0.049],
    [90600, 43.5, -0.257],
    [60300, 22.0, -0.210],
    [13000,  0.0, -0.467]
], columns=cols, index=rows)

interaction_parameters = {
    'G1984': G1984,
    'GU1986': GU1986,
    'NB1987': NB1987,
    'LN1988': LN1988,
    'FL1988': FL1988,
    'EG1990': EG1990,
    'B2004_Al': B2004_Al,
    'B2004_MM': B2004_MM
}


def calculate_margules(inter_params: pd.DataFrame, T: float, P: float) -> dict:
    """Calculate Margules Parameters (WG) for each feldspar component. Using WG = WH - T*WS + P*WV.

    Params:
        inter_params (pd.DataFrame): Thermodynamic interaction parameters (Entropy, Enthalpy and Volume) for Feldspars.
        T (float): Temperature in Kelvin.
        P (float): Pressure in Bar.

    Returns:
        WG (dict): Gibbs Free Energy Margules Parameter for each Feldspar component.
    """
    components = ["WAbOr", "WOrAb", "WAbAn", "WAnAb", "WOrAn", "WAnOr", "WAbOrAn"]

    WG = {
        comp: inter_params["WH"][comp] - ((T) * inter_params["WS"][comp]) + (P * inter_params["WV"][comp])
        for comp in components
    }

    return WG


def calculate_activity(WG: dict, X: pd.Series, T: float, component: str) -> tuple[np.ndarray]:
    """Calculate activity for a given feldspar component based on its type.

    Params:
        WG (dict): Gibbs Free Energy Margules Parameter for each Feldspar component.
        X (pd.Series): Feldspar composition.
        T (float): Temperature in Kelvin.
        component (str): Type of feldspar component ('ab', 'or', or 'an').

    Returns:
        terms (np.ndarray): Temporary Margules parameters.
        activity (float): Activity of the specified feldspar component.
    """
    terms = np.zeros(7)

    R = 8.31446261815324  # Molar Gas Constant [J⋅K−1⋅mol−1]

    if component not in ["ab", "or", "an"]:
        raise ValueError("Component not one of ab, or, or an.")

    if component == 'ab':
        terms[0] = WG["WOrAb"] * ((2 * X['Ab'] * X['Or'] * (1 - X['Ab'])) + X['Or'] * X['An'] * (0.5 - X['Ab']))
        terms[1] = WG["WAbOr"] * ((X['Or']**2) * (1 - (2 * X['Ab'])) + X['Or'] * X['An'] * (0.5 - X['Ab']))
        terms[2] = WG["WAbAn"] * ((X['An']**2) * (1 - (2 * X['Ab'])) + X['Or'] * X['An'] * (0.5 - X['Ab']))
        terms[3] = WG["WAnAb"] * ((2 * X['An'] * X['Ab'] * (1 - X['Ab'])) + X['Or'] * X['An'] * (0.5 - X['Ab']))
        terms[4] = WG["WOrAn"] * (X['Or'] * X['An'] * (0.5 - X['Ab'] - (2 * X['An'])))
        terms[5] = WG["WAnOr"] * (X['Or'] * X['An'] * (0.5 - X['Ab'] - (2 * X['Or'])))
        terms[6] = WG["WAbOrAn"] * (X['Or'] * X['An'] * (1 - (2 * X['Ab'])))
        activity = X['Ab'] * np.exp(sum(terms) / (R * T))

    elif component == 'or':
        terms[0] = WG["WOrAb"] * ((X['Ab']**2) * (1 - (2 * X['Or'])) + X['Ab'] * X['An'] * (0.5 - X['Or']))
        terms[1] = WG["WAbOr"] * ((2 * X['Ab'] * X['Or'] * (1 - X['Or'])) + X['Ab'] * X['An'] * (0.5 - X['Or']))
        terms[2] = WG["WAbAn"] * (X['Ab'] * X['An'] * (0.5 - X['Or'] - (2 * X['An'])))
        terms[3] = WG["WAnAb"] * (X['Ab'] * X['An'] * (0.5 - X['Or'] - (2 * X['Ab'])))
        terms[4] = WG["WOrAn"] * ((X['An']**2) * (1 - (2 * X['Or'])) + X['Ab'] * X['An'] * (0.5 - X['Or']))
        terms[5] = WG["WAnOr"] * ((2 * X['Or'] * X['An'] * (1 - X['Or'])) + X['Ab'] * X['An'] * (0.5 - X['Or']))
        terms[6] = WG["WAbOrAn"] * (X['An'] * X['Ab'] * (1 - (2 * X['Or'])))
        activity = X['Or'] * np.exp(sum(terms) / (R * T))

    elif component == 'an':
        terms[0] = WG["WOrAb"] * (X['Ab'] * X['Or'] * (0.5 - X['An'] - (2 * X['Ab'])))
        terms[1] = WG["WAbOr"] * (X['Ab'] * X['Or'] * (0.5 - X['An'] - (2 * X['Or'])))
        terms[2] = WG["WAbAn"] * (2 * X['Ab'] * X['An'] * (1 - X['An']) + X['Ab'] * X['Or'] * (0.5 - X['An']))
        terms[3] = WG["WAnAb"] * ((X['Ab']**2) * (1 - (2 * X['An'])) + X['Ab'] * X['Or'] * (0.5 - X['An']))
        terms[4] = WG["WOrAn"] * (2 * X['Or'] * X['An'] * (1 - X['An']) + X['Ab'] * X['Or'] * (0.5 - X['An']))
        terms[5] = WG["WAnOr"] * ((X['Or']**2) * (1 - (2 * X['An'])) + X['Ab'] * X['Or'] * (0.5 - X['An']))
        terms[6] = WG["WAbOrAn"] * (X['Or'] * X['Ab'] * (1 - (2 * X['An'])))
        activity = X['An'] * np.exp(sum(terms) / (R * T))

    return terms, activity


def calculate_temp(
    Af_X: pd.Series,
    Pf_X: pd.Series,
    P: float,
    T_init: float,
    inter_params: pd.DataFrame,
    component: str
) -> float:
    """Calculate ternary feldspar component temperature.

    Params:
        Af_X (pd.Series): Alkali feldspar composition.
        Pf_X (pd.Series): Plagioclase feldspar composition.
        P (float): Pressure in kbar.
        T_init (float): Initial guess temperature in Kelvin.
        inter_params (pd.DataFrame): Ternary Feldspar thermodynamic interaction parameters.
        component (str): Type of feldspar component ('ab', 'or', or 'an').

    Returns:
        result (scipy.Result): Minimisation result.
    """
    def obj(T):  # Objective function to minimize
        G = calculate_margules(inter_params, T, P)
        W_af, a_Af = calculate_activity(G, Af_X, T, component)
        W_pf, a_Pf = calculate_activity(G, Pf_X, T, component)
        return 1E5*np.abs(a_Af - a_Pf)

    result = optimize.minimize_scalar(obj, bounds=(500, 1500), method='bounded')
    return result.x if result.success else np.nan


def tf_temp(
    Af_X: pd.DataFrame | pd.Series,
    Pf_X: pd.DataFrame | pd.Series,
    P: float | list[float],
    T_init: float,
    inter_params: pd.DataFrame
) -> pd.DataFrame:
    """Calculate ternary feldspar temperature for multiple rows of compositions.

    Params:
        Af_X (pd.DataFrame | pd.Series): Alkali feldspar composition.
        Pf_X (pd.DataFrame | pd.Series): Plagioclase feldspar composition.
        P (float): Pressure in kbar.
        T_init (float): Initial guess temperature in Kelvin.
        inter_params (pd.DataFrame): Ternary Feldspar thermodynamic parameters.

    Returns:
        Results (pd.DataFrame): Calculated temperatures in Kelvin and Celsius and statistics for each af_row.
    """
    # Check if Af_X and Pf_X are Series and convert to DataFrame if necessary
    if isinstance(Af_X, pd.Series):
        Af_X = Af_X.to_frame().T
    if isinstance(Pf_X, pd.Series):
        Pf_X = Pf_X.to_frame().T

    if isinstance(P, float) or isinstance(P, int):
        P = P * np.ones(len(Af_X))

    results = []

    for index, af_row in Af_X.iterrows():
        # Retrieve the corresponding af_row in Pf_X based on the index
        pf_row = Pf_X.loc[index]

        # Calculate temperatures
        Ab_T_K = calculate_temp(af_row, pf_row, P[index], T_init, inter_params, component="ab")
        An_T_K = calculate_temp(af_row, pf_row, P[index], T_init, inter_params, component="an")
        Or_T_K = calculate_temp(af_row, pf_row, P[index], T_init, inter_params, component="or")

        # Convert temperatures to Celsius
        Ab_T_C = Ab_T_K - 273.15
        An_T_C = An_T_K - 273.15
        Or_T_C = Or_T_K - 273.15

        # Calculate statistics
        Bar_T_C = (abs(Ab_T_C - An_T_C) + abs(Ab_T_C - Or_T_C) + abs(An_T_C - Or_T_C))
        Mean_T_C = np.mean([Ab_T_C, An_T_C, Or_T_C])
        Std_T_C = np.std([Ab_T_C, An_T_C, Or_T_C])

        # Collect results
        results.append([
            af_row["Ab"], af_row["Or"], af_row["An"],  # Af_X values
            pf_row["Ab"], pf_row["Or"], pf_row["An"],  # Pf_X values
            P[index],
            Ab_T_K, An_T_K, Or_T_K,  # Temperatures in K
            Ab_T_C, An_T_C, Or_T_C,  # Temperatures in C
            Bar_T_C, Mean_T_C, Std_T_C  # Statistics
        ])

    # Define column names
    cols = [
        "Af_XAb", "Af_XOr", "Af_XAn",
        "Pf_XAb", "Pf_XOr", "Pf_XAn",
        "P_bar",
        "T_Ab_K", "T_An_K", "T_Or_K",
        "T_Ab_C", "T_An_C", "T_Or_C",
        "T_Bar_C", "T_Mean_C", "T_Std_C"
    ]

    return pd.DataFrame(results, columns=cols)


def feld_perturb(X: pd.Series, amount: float = 0.0025) -> pd.Series:
    """perturb feldspar composition.

    Params:
        X (pd.Series): Feldspar composition to be perturbed.

    Returns:
        X_perturb (pd.Series): perturbed compositon.
    """
    perturb = pd.DataFrame([
        [ 2, -1, -1],
        [-1, -1,  2],
        [-1,  2, -1],
        [-2,  1,  1],
        [ 1,  1, -2],
        [ 1, -2,  1],
    ], columns=["An", "Ab", "Or"])

    perturb = perturb * amount
    perturb = pd.concat([perturb, perturb*2, perturb*3])  # add stronger perturbations
    perturb = pd.concat([pd.DataFrame({"An": 0, "Ab": 0, "Or": 0}, index=[0]), perturb]).reset_index(drop=True)  # add a non-perturbation to include original comp

    return X + perturb


def feld_comb(Af_X: pd.DataFrame, Pf_X: pd.DataFrame) -> tuple[pd.DataFrame]:
    """Find unique combinations of feldspar compositions.

    Params:
        Af_X (pd.DataFrame): Alkali feldspar compositional parameters.
        Pf_X (pd.DataFrame): Plagioclase feldspar compositional parameters.

    Returns:
        Af_X_comb (pd.DataFrame): Alkali feldspar compositional combinations.
        Pf_X_comb (pd.DataFrame): Plagioclase feldspar compositional combinations.
    """
    if len(Af_X) != len(Pf_X):
        raise ValueError("Input feldspar compositions are not the same length.")

    if set(Af_X.columns) != set(Pf_X.columns):
        raise ValueError("Column headers are not the same for both feldspar compositions.")

    # Perform a cross join to produce unique af_row combinations
    Af_X['_key'], Pf_X['_key'] = 1, 1
    combined = pd.merge(Af_X, Pf_X, on='_key').drop('_key', axis=1)

    # Separate combined dataframe
    Af_X_comb = combined.iloc[:, :len(Af_X.columns)-1].copy()
    Pf_X_comb = combined.iloc[:, len(Af_X.columns)-1:].copy()
    Af_X_comb.columns = Af_X_comb.columns.str.replace('_x', '', regex=False)
    Pf_X_comb.columns = Pf_X_comb.columns.str.replace('_y', '', regex=False)

    return Af_X_comb, Pf_X_comb


def tf_temp_perturb(
    Af_X: pd.DataFrame | pd.Series,
    Pf_X: pd.DataFrame | pd.Series,
    P: float | list[float],
    T_init: float,
    inter_params: pd.DataFrame
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Calculate ternary feldspar temperature by perturbing feldspar composition. Uses same approach as Furhman and Lindsley 1988.
    Produces the best perturbed composition and temperature by finding that which has the smallest standard deviation between the three calculated temperatures.

    Params:
        Af_X (pd.DataFrame | pd.Series): Alkali feldspar composition.
        Pf_X (pd.DataFrame | pd.Series): Plagioclase feldspar composition.
        P (float): Pressure in kbar.
        T_init (float): Initial guess temperature in Kelvin.
        inter_params (pd.DataFrame): Ternary Feldspar thermodynamic parameters.

    Returns:
        Best (pd.DataFrame): Composition with most similar ternary temperatures.
        All (pd.DataFrame): All calculated temperatures from perturbed compositions.
    """
    Best = pd.DataFrame(columns=[
        "Af_XAb", "Af_XOr", "Af_XAn",
        "Pf_XAb", "Pf_XOr", "Pf_XAn",
        "P_bar",
        "T_Ab_K", "T_An_K", "T_Or_K",
        "T_Ab_C", "T_An_C", "T_Or_C",
        "T_Bar_C", "T_Mean_C", "T_Std_C"
    ])

    All = []

    # Check if Af_X and Pf_X are Series and convert to DataFrame if necessary
    if isinstance(Af_X, pd.Series):
        Af_X = Af_X.to_frame().T
    if isinstance(Pf_X, pd.Series):
        Pf_X = Pf_X.to_frame().T

    if isinstance(P, float) or isinstance(P, int):
        P = P * np.ones(len(Af_X))

    for index, af_row in Af_X.iterrows():
        pf_row = Pf_X.loc[index]

        # perturb Feldspar Compositions
        Af_perturb = feld_perturb(af_row)
        Pf_perturb = feld_perturb(pf_row)

        # Find Unique Combinations
        Af_comb, Pf_comb = feld_comb(Af_perturb, Pf_perturb)

        # Calculate pair temperatures
        Comb_T = tf_temp(Af_comb, Pf_comb, P[index], T_init, inter_params)
        Comb_T = Comb_T.sort_values("T_Std_C", ascending=True).reset_index(drop=True)

        Best = pd.concat([Best, Comb_T.loc[0].to_frame().T])
        All.append(Comb_T)

    return Best, All