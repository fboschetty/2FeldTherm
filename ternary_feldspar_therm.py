import numpy as np
import pandas as pd
from scipy import optimize
from dataclasses import dataclass, field


# CONSTANTS
R = 8.31446261815324  # Molar Gas Constant [J⋅K−1⋅mol−1]


@dataclass
class interaction_parameters:
    """Class to manage multiple parameter sets and their associated study names."""

    # Define common columns and row names as class attributes
    COLUMNS: list = field(default_factory=lambda: ["WH", "WS", "WV"])
    ROWS: list = field(default_factory=lambda: [
        "WAbOr", "WOrAb", "WAbAn", "WAnAb", "WOrAn", "WAnOr", "WAbOrAn"
    ])

    _parameters: dict = field(default_factory=dict)

    def add_parameters(self, key: str, data: list, study_name: str):
        """Add a parameter set with a specified key, data, and study name."""
        self._parameters[key] = {
            'parameters': self.create_dataframe(data),
            'study_name': study_name
        }

    def create_dataframe(self, data: list) -> pd.DataFrame:
        """Create a DataFrame from the provided data and common column/row names."""
        return pd.DataFrame(data, columns=self.COLUMNS, index=self.ROWS)

    def get_parameters(self, key: str) -> pd.DataFrame:
        """Return the parameters DataFrame for the specified key."""
        return self._parameters[key]['parameters'] if key in self._parameters else None

    def get_study(self, key: str) -> str:
        """Return the study name for the specified key."""
        return self._parameters[key]['study_name'] if key in self._parameters else None

    def display_all(self):
        """Display all keys with their corresponding study names."""
        print(f"Available Interaction Parameters")
        for key, value in self._parameters.items():
            print(f"{key}: {value['study_name']}")

    def __str__(self):
        """Return a string representation of all parameter set keys."""
        return f"interaction_parameters(keys={list(self._parameters.keys())})"

# Create an instance of interaction_parameters
interaction_parameters = interaction_parameters()

# Add parameter sets for each study
interaction_parameters.add_parameters(
    key='G1984',
    data=[
        [30978, 21.40, 0.361],
        [17062, 0.0, 0.361],
        [28226, 0.0, 0.0],
        [8741, 0.0, 0.0],
        [67469, -20.21, 0.0],
        [27983, 11.06, 0.0],
        [-13869, -14.63, 0.0]
    ],
    study_name="Ghiorso 1984"
)

interaction_parameters.add_parameters(
    key='GU1986',
    data=[
        [18810, 10.3, 0.364],
        [27320, 10.3, 0.364],
        [28226, 0.0, 0.0],
        [8743, 0.0, 0.0],
        [65305, -114.104, 0.9699],
        [-65407, 12.537, 2.1121],
        [0.0, 0.0, -1.094]
    ],
    study_name="Green and Usdansky 1986"
)

interaction_parameters.add_parameters(
    key='NB1987',
    data=[
        [30978.0, 21.40, 0.361],
        [17062.0, 0.0, 0.361],
        [14129.4, 6.14, 0.0],
        [11225.7, 7.87, 0.0],
        [25030.3, -10.80, 0.0],
        [75023.3, 22.97, 0.0],
        [0.0, 0.0, 0.0]
    ],
    study_name="Nekvasil and Burnham 1987"
)

interaction_parameters.add_parameters(
    key='LN1988',
    data=[
        [18810, 10.30, 0.4602],
        [27320, 10.30, 0.3264],
        [14129, 6.18, 0.0],
        [11226, 7.87, 0.0],
        [33415, 0.0, 0.0],
        [43369, 8.43, -0.1037],
        [19969, 0.0, -1.0950]
    ],
    study_name="Lindsley and Nekvasil 1988"
)

interaction_parameters.add_parameters(
    key='FL1988',
    data=[
        [18810, 10.3, 0.394],
        [27320, 10.3, 0.394],
        [28226, 0.0, 0.0],
        [8741, 0.0, 0.0],
        [52468, 0.0, 0.0],
        [47396, 0.0, -0.120],
        [8700, 0.0, -1.094]
    ],
    study_name="Fuhrman and Lindsley 1988"
)

interaction_parameters.add_parameters(
    key='EG1990',
    data=[
        [18810, 10.3, 0.4602],
        [27320, 10.3, 0.3264],
        [7924, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [40317, 0.0, 0.0],
        [38974, 0.0, -0.1037],
        [12545, 0.0, -1.095]
    ],
    study_name="Elkins and Grove 1990"
)

interaction_parameters.add_parameters(
    key='B2004_Al',
    data=[
        [19550, 10.5, 0.327],
        [22820, 6.3, 0.461],
        [31000, 4.5, 0.069],
        [9800, -1.7, -0.049],
        [90600, 29.5, -0.257],
        [60300, 11.2, -0.210],
        [8000, 0.0, -0.467]
    ],
    study_name="Benisek et al 2004 Aluminium Avoidance"
)

interaction_parameters.add_parameters(
    key='B2004_MM',
    data=[
        [19550, 10.5, 0.327],
        [22820, 6.3, 0.461],
        [31000, 19.0, 0.069],
        [9800, 7.5, -0.049],
        [90600, 43.5, -0.257],
        [60300, 22.0, -0.210],
        [13000, 0.0, -0.467]
    ],
    study_name="Benisek et al 2004 Molecular Mixing"
)


def calculate_margules(inter_params: pd.DataFrame, T: float, P: float) -> dict:
    """Calculate Margules Parameters (WG) for each feldspar component. Using WG = WH - T*WS + P*WV.

    Params:
        inter_params (pd.DataFrame): Thermodynamic interaction parameters (Entropy, Enthalpy and Volume) for Feldspars.
        T (float): Temperature in Kelvin.
        P (float): Pressure in Bar.

    Returns:
        WG (dict): Gibbs Free Energy Margules Parameter for each Feldspar component.
    """

    return inter_params["WH"] - T * inter_params["WS"] + P * inter_params["WV"]


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
    if component not in ["ab", "or", "an"]:
        raise ValueError("Component not one of ab, or, or an.")

    terms = np.zeros(7)

    X_Ab, X_Or, X_An = X["Ab"], X["Or"], X["An"]

    if component == 'ab':
        terms[0] = WG["WOrAb"] * ((2 * X_Ab * X_Or * (1 - X_Ab)) + X_Or * X_An * (0.5 - X_Ab))
        terms[1] = WG["WAbOr"] * ((X_Or**2) * (1 - (2 * X_Ab)) + X_Or * X_An * (0.5 - X_Ab))
        terms[2] = WG["WAbAn"] * ((X_An**2) * (1 - (2 * X_Ab)) + X_Or * X_An * (0.5 - X_Ab))
        terms[3] = WG["WAnAb"] * ((2 * X_An * X_Ab * (1 - X_Ab)) + X_Or * X_An * (0.5 - X_Ab))
        terms[4] = WG["WOrAn"] * (X_Or * X_An * (0.5 - X_Ab - (2 * X_An)))
        terms[5] = WG["WAnOr"] * (X_Or * X_An * (0.5 - X_Ab - (2 * X_Or)))
        terms[6] = WG["WAbOrAn"] * (X_Or * X_An * (1 - (2 * X_Ab)))
        activity = X_Ab * np.exp(sum(terms) / (R * T))

    elif component == 'or':
        terms[0] = WG["WOrAb"] * ((X_Ab**2) * (1 - (2 * X_Or)) + X_Ab * X_An * (0.5 - X_Or))
        terms[1] = WG["WAbOr"] * ((2 * X_Ab * X_Or * (1 - X_Or)) + X_Ab * X_An * (0.5 - X_Or))
        terms[2] = WG["WAbAn"] * (X_Ab * X_An * (0.5 - X_Or - (2 * X_An)))
        terms[3] = WG["WAnAb"] * (X_Ab * X_An * (0.5 - X_Or - (2 * X_Ab)))
        terms[4] = WG["WOrAn"] * ((X_An**2) * (1 - (2 * X_Or)) + X_Ab * X_An * (0.5 - X_Or))
        terms[5] = WG["WAnOr"] * ((2 * X_Or * X_An * (1 - X_Or)) + X_Ab * X_An * (0.5 - X_Or))
        terms[6] = WG["WAbOrAn"] * (X_An * X_Ab * (1 - (2 * X_Or)))
        activity = X_Or * np.exp(sum(terms) / (R * T))

    elif component == 'an':
        terms[0] = WG["WOrAb"] * (X_Ab * X_Or * (0.5 - X_An - (2 * X_Ab)))
        terms[1] = WG["WAbOr"] * (X_Ab * X_Or * (0.5 - X_An - (2 * X_Or)))
        terms[2] = WG["WAbAn"] * (2 * X_Ab * X_An * (1 - X_An) + X_Ab * X_Or * (0.5 - X_An))
        terms[3] = WG["WAnAb"] * ((X_Ab**2) * (1 - (2 * X_An)) + X_Ab * X_Or * (0.5 - X_An))
        terms[4] = WG["WOrAn"] * (2 * X_Or * X_An * (1 - X_An) + X_Ab * X_Or * (0.5 - X_An))
        terms[5] = WG["WAnOr"] * ((X_Or**2) * (1 - (2 * X_An)) + X_Ab * X_Or * (0.5 - X_An))
        terms[6] = WG["WAbOrAn"] * (X_Or * X_Ab * (1 - (2 * X_An)))
        activity = X_An * np.exp(sum(terms) / (R * T))

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
        _, a_Af = calculate_activity(G, Af_X, T, component)
        _, a_Pf = calculate_activity(G, Pf_X, T, component)
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
    # Create P array if single value provided
    if isinstance(P, (float, int)):
        P = np.full(len(Af_X), P)

    # Pre-allocate results arrays
    Ab_T_K = np.empty(len(Af_X))
    An_T_K = np.empty(len(Af_X))
    Or_T_K = np.empty(len(Af_X))

    # Calculate temperatures for all rows using vectorized operations
    for index in range(len(Af_X)):
        af_row = Af_X.iloc[index]
        pf_row = Pf_X.iloc[index]

        Ab_T_K[index] = calculate_temp(af_row, pf_row, P[index], T_init, inter_params, 'ab')
        An_T_K[index] = calculate_temp(af_row, pf_row, P[index], T_init, inter_params, 'an')
        Or_T_K[index] = calculate_temp(af_row, pf_row, P[index], T_init, inter_params, 'or')

    # Create results DataFrame
    results_df = pd.DataFrame({
        "Af_XAb": Af_X["Ab"],
        "Af_XOr": Af_X["Or"],
        "Af_XAn": Af_X["An"],
        "Pf_XAb": Pf_X["Ab"],
        "Pf_XOr": Pf_X["Or"],
        "Pf_XAn": Pf_X["An"],
        "P_bar": P,
        "T_Ab_K": Ab_T_K,
        "T_An_K": An_T_K,
        "T_Or_K": Or_T_K
    })

    # Convert temperatures from Kelvin to Celsius
    results_df[["T_Ab_C", "T_An_C", "T_Or_C"]] = results_df[["T_Ab_K", "T_An_K", "T_Or_K"]] - 273.15

    # Calculate statistics (temperature differences, mean, and standard deviation)
    results_df["T_Bar_C"] = np.abs(results_df["T_Ab_C"] - results_df["T_An_C"]) + \
                            np.abs(results_df["T_Ab_C"] - results_df["T_Or_C"]) + \
                            np.abs(results_df["T_An_C"] - results_df["T_Or_C"])

    results_df["T_Mean_C"] = results_df[["T_Ab_C", "T_An_C", "T_Or_C"]].mean(axis=1)
    results_df["T_Std_C"] = results_df[["T_Ab_C", "T_An_C", "T_Or_C"]].std(axis=1, ddof=0)

    return results_df


def feld_perturb(X: pd.Series, amount: float = 0.0025) -> pd.Series:
    """perturb feldspar composition.

    Params:
        X (pd.Series): Feldspar composition to be perturbed.

    Returns:
        X_perturb (pd.Series): perturbed compositon.
    """
    base_perturb = np.array([
        [ 2, -1, -1],
        [-1, -1,  2],
        [-1,  2, -1],
        [-2,  1,  1],
        [ 1,  1, -2],
        [ 1, -2,  1],
    ])

    perturb_levels = [base_perturb * i for i in range(1, 4)]  # Make two more levels of 'stronger' perturbation
    all_perturb = np.vstack([np.zeros((1, 3)), *perturb_levels]) * amount  # Add row of zeros to include original composition
    perturb_df = pd.DataFrame(all_perturb, columns = ["An", "Ab", "Or"])  # Convert to df to avoid mixing up columns

    return X + perturb_df


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

    # Efficient cross join
    Af_X_comb = pd.concat([Af_X] * len(Pf_X), ignore_index=True)
    Pf_X_comb = pd.concat([Pf_X] * len(Af_X), ignore_index=True)

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
    # Lists to store each set of results
    Best = []
    All = []

    # Check if Af_X and Pf_X are Series and convert to DataFrame if necessary
    if isinstance(Af_X, pd.Series):
        Af_X = Af_X.to_frame().T
    if isinstance(Pf_X, pd.Series):
        Pf_X = Pf_X.to_frame().T
    # Create Pressure array if it's a single value
    if isinstance(P, (float, int)):
        P = np.full(len(Af_X), P)

    # Calculate temperatures for all perturbations
    for index in range(len(Af_X)):
        af_row = Af_X.iloc[index]
        pf_row = Pf_X.iloc[index]

        # Perturb Feldspar Compositions
        Af_perturb = feld_perturb(af_row)
        Pf_perturb = feld_perturb(pf_row)

        # Find Unique Combinations
        Af_comb, Pf_comb = feld_comb(Af_perturb, Pf_perturb)

        # Calculate pair temperatures
        Comb_T = tf_temp(Af_comb, Pf_comb, P[index], T_init, inter_params)

        # Store the result and the best row
        All.append(Comb_T)  # Store all results
        min_idx = Comb_T['T_Std_C'].idxmin()  # Find the index of the minimum
        Best.append(Comb_T.loc[min_idx])  # Append the best result

    # Convert Best results to DataFrame
    Best_df = pd.DataFrame(Best).reset_index(drop=True)

    return Best_df, All