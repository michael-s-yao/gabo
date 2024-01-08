"""
IWPC warfarin dataset implementation.

Author(s):
    Michael Yao @michael-s-yao

Data preprocessing logic adapted from @gianlucatruda at
https://github.com/gianlucatruda/warfit-learn/tree/master

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Sequence, Union


class WarfarinDataset:
    def __init__(self, root: Union[Path, str] = "./data/warfarin"):
        """
        Args:
            root: directory path. Default `./warfarin/data`.
        """
        self.root = root

        self.height = "Height (cm)"
        self.weight = "Weight (kg)"
        self.age = "Age"
        self.race = "Race (OMB)"
        self.gender = "Gender"
        self.CYP2C9 = "CYP2C9 consensus"
        self.VKORC1 = "Imputed VKORC1"
        self.outcome = "INR on Reported Therapeutic Dose of Warfarin"
        self.rare_alleles = ["*1/*5", "*1/*6", "*1/*11", "*1/*13", "*1/*14"]
        self.dose = "Therapeutic Dose of Warfarin"
        self.did_reach_stable_dose = "Subject Reached Stable Dose of Warfarin"
        self.inr = "INR on Reported Therapeutic Dose of Warfarin"
        self.target_inr = "Estimated Target INR Range Based on Indication"
        self.thresh = 315  # Threshold for extreme weekly warfarin dose.

        with open(os.path.join(self.root, "metadata.json"), "rb") as f:
            self.metadata = json.load(f)
        with pd.ExcelFile(
            os.path.join(self.root, self.metadata["dataset"])
        ) as xls:
            self.dataset = pd.read_excel(xls, "Subject Data")[self.columns]

        self._prune()
        self._impute_heights_and_weights()
        self.dataset = self.dataset.fillna(0)

        self.patients = self.dataset.drop(
            [self.inr, self.target_inr, self.did_reach_stable_dose], axis=1
        )
        self.doses = self.dataset[self.dose]

        self.transform = StandardizeTransform(
            self.dataset, self.height, self.weight, self.dose
        )

    def _prune(self) -> None:
        """
        Cleans up the initial dataset.
        Input:
            None.
        Returns:
            None.
        """
        # Drop any patients where both the height and weight are missing.
        self.dataset.dropna(
            subset=[self.height, self.weight], how="all", inplace=True
        )
        # Drop any patients where either the race or gender are missing.
        self.dataset.dropna(
            subset=[self.race, self.gender, self.age], how="any", inplace=True
        )
        # Convert ages to decades.
        self.dataset[self.age] = np.array(
            [int(x[0]) for x in self.dataset[self.age]]
        )
        # Impute target INRs.
        self._impute_target_INR()
        # Impute VKORC1 SNPs.
        self.dataset[self.VKORC1] = self.dataset.apply(
            self._impute_VKORC1, axis=1
        )
        old_VKORC1 = [
            "VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G",
            "VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G",
            "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T",  # noqa
            "VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G"
        ]
        self.dataset.drop(old_VKORC1, axis=1, inplace=True)
        # Convert the race and gender variables to one-hot encodings.
        self.dataset = pd.get_dummies(self.dataset, columns=[self.race])
        self.dataset["Gender"] = self.dataset["Gender"] == "male"
        self.races = [
            key for key in self.dataset.keys() if key.startswith(self.race)
        ]
        # Drop unuseable rows that are missing essential data points.
        self.dataset.dropna(
            subset=[
                self.dose,
                self.outcome,
                "Subject Reached Stable Dose of Warfarin",
                self.CYP2C9
            ],
            inplace=True
        )
        # Exclude rare alleles of the CYP2C9 gene.
        valid_idxs = self.dataset[self.CYP2C9].isin(self.rare_alleles) == False
        self.dataset = self.dataset[valid_idxs]
        # Convert NaN CYP2C9 genotypes to "Unknown".
        self.dataset.loc[
            self.dataset[self.CYP2C9].isna(), self.CYP2C9
        ] = "Unknown"
        # Exclude extreme warfarin doses.
        self.dataset = self.dataset[self.dataset[self.dose] < self.thresh]
        # Convert the remaining categorical variables to one-hot encodings.
        self.dataset = pd.get_dummies(
            self.dataset, columns=[self.CYP2C9, self.VKORC1]
        )

    def _impute_heights_and_weights(self) -> None:
        """
        Impute missing height values using weight, race, and sex. Impute
        missing weight values using height, race, and sex. Both imputations
        are performed using linear regression.
        Input:
            None.
        Returns:
            None.
        """
        train = self.dataset[
            (self.dataset[self.height].isnull() == False) & (
                self.dataset[self.weight].isnull() == False
            )
        ]

        pred_h = self.dataset[self.dataset[self.height].isnull()]
        pred_h = pred_h.drop(self.height, axis=1)
        pred_h = pred_h[[self.weight] + self.races + [self.gender]]
        X_h, y_h = train.drop(self.height, axis=1), train[self.height]
        X_h = X_h[[self.weight] + self.races + [self.gender]]
        model_h = LinearRegression()
        model_h.fit(X_h, y_h)

        pred_w = self.dataset[self.dataset[self.weight].isnull()]
        pred_w = pred_w.drop(self.weight, axis=1)
        pred_w = pred_w[[self.height] + self.races + [self.gender]]
        X_w, y_w = train.drop(self.weight, axis=1), train[self.weight]
        X_w = X_w[[self.height] + self.races + [self.gender]]
        model_w = LinearRegression()
        model_w.fit(X_w, y_w)

        self.dataset.loc[self.dataset[self.height].isnull(), self.height] = (
            model_h.predict(pred_h)
        )
        self.dataset.loc[self.dataset[self.weight].isnull(), self.weight] = (
            model_w.predict(pred_w)
        )

        return

    def _impute_target_INR(self, consensus_target_inr: float = 2.5) -> None:
        """
        Impute target INRs using the mean of a range if an INR range is given,
        and using the consensus INR target if no information is given.
        Input:
            consensus_target_inr: consensus INR target default value.
        Returns:
            None.
        """
        target_inr = self.dataset[self.target_inr]
        target_inr = target_inr.fillna(consensus_target_inr)
        imputed = []
        for i, inr in enumerate(target_inr):
            if isinstance(inr, str):
                low, high = inr.replace(" ", "").split("-")
                inr = (float(low) + float(high)) / 2.0
            imputed.append(inr)
        self.dataset[self.target_inr] = np.array(imputed)
        return

    def _impute_VKORC1(self, datum: pd.Series) -> str:
        """
        Impute VKORC1 SNPs using the method from Klein et al. (2009).
        Input:
            datum: a single patient data point from the dataset.
        Returns:
            Imputed VKORC1 SNP for the patient.
        Citation(s):
            [1] The International Warfarin Pharmacogenetics Consortium.
                Estimation of the warfarin dose with clinical and
                pharmacogenetic data. N Engl J Med 360:753-64. (2009).
                https://doi.org/10.1056/NEJMoa0809329
        """
        rs2359612 = datum[
            "VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G"
        ]
        rs9934438 = datum[
            "VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G"
        ]
        rs9923231 = datum[
            "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"
        ]
        rs8050894 = datum[
            "VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G"
        ]
        race = datum[self.race]
        black_missing_mixed = [
            "Black or African American", "Missing or Mixed Race"
        ]

        if rs9923231 in ["A/A", "A/G", "G/A", "G/G"]:
            return rs9923231
        elif race not in black_missing_mixed and rs2359612 == "C/C":
            return "G/G"
        elif race not in black_missing_mixed and rs2359612 == "T/T":
            return "A/A"
        elif rs9934438 == "C/C":
            return "G/G"
        elif rs9934438 == "T/T":
            return "A/A"
        elif rs9934438 == "C/T":
            return "A/G"
        elif race not in black_missing_mixed and rs8050894 == "G/G":
            return "G/G"
        elif race not in black_missing_mixed and rs8050894 == "C/C":
            return "A/A"
        elif race not in black_missing_mixed and rs8050894 == "C/G":
            return "A/G"
        else:
            return "Unknown"

    @property
    def columns(self) -> Sequence[str]:
        """
        Returns the variables of interest from the dataset.
        Input:
            None.
        Returns:
            Variables of interest from the IWPC warfarin dataset.
        """
        return [
            "Gender",
            "Race (OMB)",
            "Age",
            "Height (cm)",
            "Weight (kg)",
            "Acetaminophen or Paracetamol (Tylenol)",
            "Simvastatin (Zocor)",
            "Atorvastatin (Lipitor)",
            "Fluvastatin (Lescol)",
            "Lovastatin (Mevacor)",
            "Pravastatin (Pravachol)",
            "Rosuvastatin (Crestor)",
            "Amiodarone (Cordarone)",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Sulfonamide Antibiotics",
            "Macrolide Antibiotics",
            "Estimated Target INR Range Based on Indication",
            "Subject Reached Stable Dose of Warfarin",
            "Therapeutic Dose of Warfarin",
            "INR on Reported Therapeutic Dose of Warfarin",
            "Current Smoker",
            "CYP2C9 consensus",
            "VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G",
            "VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G",
            "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T",  # noqa
            "VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G"
        ]


class StandardizeTransform:
    def __init__(self, X: pd.DataFrame, *args):
        """
        Args:
            X: an input DataFrame of patient data.
            args: the column names to standardize.
        """
        self.args = args
        self.transforms = {
            col: StandardScaler().fit(X.loc[:, col].to_numpy().reshape(-1, 1))
            for col in self.args
        }

    def standardize(
        self, X: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Standardizes the specified columns in the input DataFrame or Series.
        Input:
            X: the DataFrame or Series of data to standardize.
        Returns:
            The DataFrame or Series of data with the standardized columns.
        """
        for col in self.args:
            if isinstance(X, pd.Series) and X.name != col:
                continue
            elif isinstance(X, pd.DataFrame) and col not in X.columns:
                continue

            if isinstance(X, pd.Series):
                return pd.Series(
                    np.squeeze(
                        self.transforms[col].transform(
                            X.to_numpy().reshape(-1, 1)
                        ),
                        axis=-1
                    )
                )
            elif isinstance(X, pd.DataFrame):
                X[col] = self.transforms[col].transform(
                    X[col].to_numpy().reshape(-1, 1)
                )

        return X

    def unstandardize(self, z: pd.DataFrame) -> pd.DataFrame:
        """
        Unstandardizes the specified columns in the input DataFrame back into
        the original representation.
        Input:
            X: the DataFrame of data to unstandardize.
        Returns:
            The DataFrame of data with the columns in their original
            representations.
        """
        for col in self.args:
            if col not in z.columns:
                continue
            z.loc[:, col] = np.squeeze(
                self.transforms[col].inverse_transform(
                    z.loc[:, col].to_numpy().reshape(-1, 1)
                ),
                axis=-1
            )
        return z
