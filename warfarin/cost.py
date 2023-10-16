"""
Defines the patient cost function as a function of the warfarin dose.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np


def dosage_cost(pred_dose: np.ndarray, gt_dose: np.ndarray) -> np.ndarray:
    """
    Defines a quadratic cost function dependent on a patient's predicted dose
    and true stable warfarin dose.
    Input:
        pred_dose: the predicted weekly warfarin dose(s) of the patient(s).
        gt_dose: the true stable weekly warfarin dose(s) of the patient(s).
    Returns:
        The cost associated with each patient's predicted dose.
    """
    return np.square(pred_dose - gt_dose)
