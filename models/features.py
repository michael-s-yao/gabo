"""
Implements a molecule fingerprints feature extractor for tokenized molecules
in the SELFIES representation space.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Krenn M, Hase F, Nigam AK, Friederich P, Aspuru-Guzik A. Self-
        referencing embedded strings (SELFIES): A 100% robust molecular string
        representation. Machine Learning: Science and Technology 1(4): 045024.
        (2020). https://doi.org/10.1088/2632-2153/aba947

Adapted from the @design-bench GitHub repository from @rail-berkeley at
https://github.com/rail-berkeley/design-bench/design_bench/oracles/
feature_extractors/morgan_fingerprint_features.py

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import sys
import numpy as np
import selfies as sf

sys.path.append(".")
import data
from design_bench.oracles.feature_extractors.morgan_fingerprint_features \
    import MorganFingerprintFeatures


class SELFIESMorganFingerprintFeatures(MorganFingerprintFeatures):
    """
    Implements a molecule fingerprints feature extractor for tokenized
    molecules in the SELFIES representation space.
    """

    def dataset_to_oracle_x(
        self, x_batch: np.ndarray, dataset: data.SELFIESChEMBLDataset
    ) -> np.ndarray:
        """
        Helper function for converting from designs contained in the SELFIES
        token dataset format into a format the oracle is expecting to process.
        Inputs:
            x_batch: a batch of input design values as SELFIES tokens.
            dataset: the dataset source of the batch.
        Returns:
            A batch of design values that have been converted into the features
            expected by the oracle score function.
        """
        features = []
        idx2vocab = {idx: tok for tok, idx in dataset.vocab2idx.items()}
        for toks in x_batch:
            sidx = np.where(toks == dataset.vocab2idx[dataset.start])[0]
            sidx = sidx[0] if sidx.size > 0 else 0
            eidx = np.where(toks == dataset.vocab2idx[dataset.stop])[0]
            eidx = eidx[0] if eidx.size > 0 else -1
            toks = toks[sidx:eidx]
            toks = [
                tk for tk in toks
                if tk not in [
                    dataset.vocab2idx[dataset.start],
                    dataset.vocab2idx[dataset.pad]
                ]
            ]

            smi = sf.decoder("".join([idx2vocab[tk] for tk in toks]))
            smi = smi.replace(" ", "")

            value = self.featurizer.featurize(smi)[0]
            features.append(
                np.zeros([self.size], dtype=self.dtype)
                if value.size != self.size
                else np.array(value, dtype=self.dtype)
            )
        return np.asarray(features)
