"""
IWPC warfarin dataset Pytorch Lightning Data Module.

Author(s):
    Michael Yao @michael-s-yao

Data preprocessing logic adapted from @gianlucatruda at
https://github.com/gianlucatruda/warfit-learn/tree/master

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from collections import defaultdict
import json
from math import isclose, isnan
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from typing import NamedTuple, Optional, Sequence, Tuple, Union


class PatientSample(NamedTuple):
    X: torch.Tensor
    X_attributes: Sequence[str]
    target_dose: float
    did_reach_stable_dose: bool
    inr: float
    target_inr: Optional[Union[float, str]]
    cost: float


class IWPCWarfarinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: Union[Path, str] = "./data/warfarin",
        train_test_split: Tuple[float] = (0.8, 0.2),
        cv_idx: Optional[int] = 0,
        batch_size: int = 128,
        num_workers: int = os.cpu_count() // 2,
        label_smoothing: Optional[float] = 0.01,
        seed: int = 42
    ):
        """
        Args:
            root: directory path. Default `./data/warfarin`.
            batch_size: batch size. Default 128.
            train_test_split: fraction of data to be allocated for each
                of the training and test splits, respectively.
            cv_idx: optional cross-validation index. Default 0.
            num_workers: number of workers. Default half the CPU count.
            label_smoothing: optional label smoothing. Default 0.01.
            seed: random seed. Default 42.
        """
        super().__init__()
        self.root = root
        self.train_frac, self.test_frac = train_test_split
        if not isclose(self.train_frac + self.test_frac, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Train and test splits must sum to 1, got {train_test_split}."
            )
        self.cv_idx = cv_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

        self.height = "Height (cm)"
        self.weight = "Weight (kg)"
        self.race = "Race (OMB)"
        self.gender = "Gender"
        self.CYP2C9 = "CYP2C9 consensus"
        self.VKORC1 = "Imputed VKORC1"
        self.intervention = "Therapeutic Dose of Warfarin"
        self.outcome = "INR on Reported Therapeutic Dose of Warfarin"
        self.rare_alleles = ["*1/*5", "*1/*6", "*1/*11", "*1/*13", "*1/*14"]
        self.other_IWPC_parameters_categorical = [
            "Age",
            "Amiodarone (Cordarone)",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Current Smoker",
            "CYP2C9 consensus",
            "Imputed VKORC1",
        ]
        self.thresh = 315  # Threshold for extreme weekly warfarin dose.

        with open(os.path.join(self.root, "metadata.json"), "rb") as f:
            self.metadata = json.load(f)
        self.dataset = pd.ExcelFile(
            os.path.join(self.root, self.metadata["dataset"])
        )
        self.dataset = self.dataset.parse("Subject Data")[self.columns]

        self._prune()

        self.num_samples, _ = self.dataset.shape
        self.rows = np.arange(self.num_samples, dtype=int)
        self.rng.shuffle(self.rows)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split dataset into train, val, and test partitions. Of the training
        dataset, we allocate 12.5% for validation. By default, this results
        in a 70% training, 10% validation, and 20% test split of the entire
        dataset.
        Input:
            stage: setup stage. One of [`fit`, `test`, None]. Default None.
        Returns:
            None.
        """
        val_frac = 0.125
        if self.cv_idx >= 1.0 / val_frac or self.cv_idx < 0:
            raise ValueError(f"Invalid cross validation index {self.cv_idx}.")
        self.num_train = round(self.train_frac * self.num_samples)
        self.num_test = self.num_samples - self.num_train
        self.num_val = round(val_frac * self.num_train)
        self.num_train = self.num_train - self.num_val

        start_val = self.cv_idx * self.num_val
        end_val = (self.cv_idx + 1) * self.num_val
        self.val = self.rows[start_val:end_val]
        self.train = np.concatenate((
            self.rows[:start_val],
            self.rows[end_val:(self.num_train + self.num_val)]
        ))
        self.test = self.rows[(self.num_train + self.num_val):]

        self.train = IWPCWarfarinDataset(
            data=self.dataset.iloc[self.train],
            columns=self.columns,
            height_key=self.height,
            weight_key=self.weight,
            race_keys=self.races,
            gender_keys=self.genders,
            model_h=None,
            model_w=None,
            label_smoothing=self.label_smoothing,
            seed=self.seed
        )
        self.val = IWPCWarfarinDataset(
            data=self.dataset.iloc[self.val],
            columns=self.columns,
            height_key=self.height,
            weight_key=self.weight,
            race_keys=self.races,
            gender_keys=self.genders,
            model_h=self.train.model_h,
            model_w=self.train.model_w,
            label_smoothing=0.0
        )

        if stage is None or stage == "test":
            self.test = IWPCWarfarinDataset(
                data=self.dataset.iloc[self.test],
                columns=self.columns,
                height_key=self.height,
                weight_key=self.weight,
                race_keys=self.races,
                gender_keys=self.genders,
                model_h=self.train.model_h,
                model_w=self.train.model_w,
                label_smoothing=0.0
            )

        return

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
            subset=[self.race, self.gender], how="any", inplace=True
        )
        # Impute VKORC1 SNPs.
        self.dataset[self.VKORC1] = self.dataset.apply(
            self.impute_VKORC1, axis=1
        )
        old_VKORC1 = [
            "VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G",
            "VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G",
            "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T",  # noqa
            "VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G"
        ]
        self.dataset.drop(old_VKORC1, axis=1, inplace=True)
        # Convert the race and gender variables to one-hot encodings.
        self.dataset = pd.get_dummies(
            self.dataset, columns=[self.race, self.gender]
        )
        self.races = [
            key for key in self.dataset.keys() if key.startswith(self.race)
        ]
        self.genders = [
            key for key in self.dataset.keys() if key.startswith(self.gender)
        ]
        # Drop unuseable rows that are missing essential data points.
        self.dataset.dropna(
            subset=[
                self.intervention,
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
        self.dataset = self.dataset[
            self.dataset[self.intervention] < self.thresh
        ]
        # Convert the remaining categorical variables to one-hot encodings.
        self.dataset = pd.get_dummies(
            self.dataset, columns=self.other_IWPC_parameters_categorical
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the training dataloader.
        Input:
            None.
        Returns:
            Training dataloader.
        """
        if self.train is None:
            return None
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the validation dataloader.
        Input:
            None.
        Returns:
            Validation dataloader.
        """
        if self.val is None:
            return None
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the test dataloader.
        Input:
            None.
        Returns:
            Test dataloader.
        """
        if self.test is None:
            return None
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch: Sequence[PatientSample]) -> PatientSample:
        """
        Defines a custom collate function.
        Input:
            batch: a sequence of patient samples to collate.
        Returns:
            The collated batch as a single PatientSample named tuple.
        """
        X, X_attributes = None, None
        for item in batch:
            if X is None:
                X = torch.unsqueeze(item.X, dim=0)
                X_attributes = item.X_attributes
            else:
                X = torch.cat((X, torch.unsqueeze(item.X, dim=0)), dim=0)
        return PatientSample(
            X,
            np.concatenate(
                (np.array(X_attributes)[np.newaxis, ...],) * len(batch), axis=0
            ),
            torch.tensor([item.target_dose for item in batch]).to(torch.float),
            torch.tensor([item.did_reach_stable_dose for item in batch]).to(
                torch.float
            ),
            torch.tensor([item.inr for item in batch]).to(torch.float),
            [item.target_inr for item in batch],
            torch.tensor([item.cost for item in batch]).to(torch.float)
        )

    def impute_VKORC1(self, datum: pd.Series) -> str:
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
            "Cerivastatin (Baycol)",
            "Amiodarone (Cordarone)",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Sulfonamide Antibiotics",
            "Macrolide Antibiotics",
            "Target INR",
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


class IWPCWarfarinDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        columns: Sequence[str],
        height_key: str,
        weight_key: str,
        race_keys: str,
        gender_keys: str,
        model_h: Optional[LinearRegression] = None,
        model_w: Optional[LinearRegression] = None,
        label_smoothing: Optional[float] = 0.0,
        seed: int = 42
    ):
        """
        Args:
            data: dataframe containing the dataset.
            columns: variables of interest from the dataset.
            height_key: label for the column corresponding to height.
            weight_key: label for the column corresponding to weight.
            race_keys: labels for the one-hot encoded columns corresponding to
                race.
            gender_keys: labels for the one-hot encoded columns corresponding
                to gender.
            model_h: optional fitted linear regression model to use to perform
                height imputation. By default, the model is fitted on the
                dataset.
            model_w: optional fitted linear regression model to use to perform
                weight imputation. By default, the model is fitted on the
                dataset.
            label_smoothing: optional label smoothing parameter.
            seed: random seed. Default 42.
        """
        super().__init__()
        self.data = data
        self.columns = columns
        self.height = height_key
        self.weight = weight_key
        self.races = race_keys
        self.genders = gender_keys
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

        self.model_h, self.model_w = self._impute_heights_and_weights(
            model_h, model_w
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        Input:
            None.
        Returns:
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> PatientSample:
        """
        Retrieves a specified item from the dataset.
        Input:
            idx: the index of the element to retrieve.
        Returns:
            A PatientSample named tuple.
        """
        pt = self.data.iloc[idx]
        target_dose = pt["Therapeutic Dose of Warfarin"]
        did_reach_stable_dose = pt["Subject Reached Stable Dose of Warfarin"]
        inr = pt["INR on Reported Therapeutic Dose of Warfarin"]
        target_inr = pt["Estimated Target INR Range Based on Indication"]
        if not isinstance(target_inr, str) and isnan(target_inr):
            target_inr = pt["Target INR"]
            if isnan(target_inr):
                target_inr = None
        pt = pt.drop([
            "Subject Reached Stable Dose of Warfarin",
            "INR on Reported Therapeutic Dose of Warfarin",
            "Estimated Target INR Range Based on Indication",
            "Target INR"
        ])
        pt = self._label_smoothing(pt.fillna(value=0)).astype(np.float32)
        return PatientSample(
            torch.tensor(pt.to_numpy()),
            pt.index.to_list(),
            target_dose,
            did_reach_stable_dose,
            inr,
            target_inr,
            IWPCWarfarinDataset.cost(inr, target_inr)
        )

    @staticmethod
    def cost(
        inr: float,
        target_inr: Optional[Union[float, str]] = None,
        consensus_target_inr: float = 2.5
    ) -> float:
        """
        Computes the cost incurred by the patient as a function of their INR.
        Per Oden and Fahlen (2002), the risk of death is approximately a
        quadratic function of INR. Therefore, we model the cost function as a
        simple quadratic function valid up to a multiplicative constant.
        Input:
            inr: patient INR.
            target_inr: optional patient target INR.
            consensus_target_inr: if `target_inr` is None, then this target_inr
                value is used as the patient's target INR.
        Returns:
            Cost incurred by the patient proportional to their risk of death.
        Citation(s):
            [1] Oden A and Fahlen M. Oral anticoagulation and risk of death: A
                medical record linkage study. BMJ 325(7372):1073-5. (2002).
                https://doi.org/10.1136%2Fbmj.325.7372.1073
        """
        if target_inr is None:
            return (inr - consensus_target_inr) ** 2
        elif isinstance(target_inr, float):
            return (inr - target_inr) ** 2
        elif isinstance(target_inr, str):
            target_inr = target_inr.replace(" ", "")
            low, high = target_inr.split("-")
            target_inr = 0.5 * (float(low) + float(high))
            return (inr - target_inr) ** 2
        else:
            raise ValueError(f"target_inr is invalid type {type(target_inr)}")

    def _impute_heights_and_weights(
        self,
        model_h: Optional[LinearRegression] = None,
        model_w: Optional[LinearRegression] = None
    ) -> Sequence[LinearRegression]:
        """
        Impute missing height values using weight, race, and sex. Impute
        missing weight values using height, race, and sex. Both imputations
        are performed using linear regression.
        Input:
            model_h: optional fitted linear regression model to use to perform
                height imputation. By default, the model is fitted on the
                dataset.
            model_w: optional fitted linear regression model to use to perform
                weight imputation. By default, the model is fitted on the
                dataset.
        Returs:
            model_h: fitted linear regression model used for height imputation.
            model_w: fitted linear regression model used for weight imputation.
        """
        train = self.data[
            (self.data[self.height].isnull() == False) & (
                self.data[self.weight].isnull() == False
            )
        ]

        pred_h = self.data[self.data[self.height].isnull()]
        pred_h = pred_h.drop(self.height, axis=1)
        pred_h = pred_h[[self.weight] + self.races + self.genders]
        if model_h is None:
            X_h, y_h = train.drop(self.height, axis=1), train[self.height]
            X_h = X_h[[self.weight] + self.races + self.genders]
            model_h = LinearRegression()
            model_h.fit(X_h, y_h)

        pred_w = self.data[self.data[self.weight].isnull()]
        pred_w = pred_w.drop(self.weight, axis=1)
        pred_w = pred_w[[self.height] + self.races + self.genders]
        if model_w is None:
            X_w, y_w = train.drop(self.weight, axis=1), train[self.weight]
            X_w = X_w[[self.height] + self.races + self.genders]
            model_w = LinearRegression()
            model_w.fit(X_w, y_w)

        self.data.loc[self.data[self.height].isnull(), self.height] = (
            model_h.predict(pred_h)
        )
        self.data.loc[self.data[self.weight].isnull(), self.weight] = (
            model_w.predict(pred_w)
        )

        return model_h, model_w

    def _label_smoothing(self, patient: pd.Series) -> pd.Series:
        """
        Performs optional label_smoothing on one-hot encoded variables.
        Input:
            patient: an input patient dataset.
        Returns:
            The original patient dataset with label smoothing applied.
        """
        if self.label_smoothing == 0.0:
            return patient
        categories = [
            "Acetaminophen",
            "Simvastatin",
            "Atorvastatin",
            "Fluvastatin",
            "Lovastatin",
            "Pravastatin",
            "Rosuvastatin",
            "Cerivastatin",
            "Sulfonamide Antibiotics",
            "Macrolide Antibiotics",
            "Race (OMB)",
            "Gender",
            "Age",
            "Amiodarone",
            "Carbamazepine",
            "Phenytoin",
            "Rifampin or Rifampicin",
            "Current Smoker",
            "CYP2C9 consensus",
            "Imputed VKORC1"
        ]
        eps = self.rng.uniform(0, self.label_smoothing, size=len(categories))
        num_classes = defaultdict(lambda: 0)
        for label in patient.index:
            for c in categories:
                if label.startswith(c):
                    num_classes[c] += 1
                    break
        for key, val in num_classes.items():
            if val <= 1:
                num_classes[key] = val + 1
        for label in patient.index:
            for c in categories:
                if label.startswith(c):
                    if int(patient[label]) > 0:
                        patient[label] -= eps[categories.index(c)]
                    else:
                        k = num_classes[c]
                        patient[label] += eps[categories.index(c)] / (k - 1)
                    break
        return patient
