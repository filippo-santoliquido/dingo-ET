import ast
from typing import Dict, Union

import h5py
import numpy as np
import pandas as pd
import scipy
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import Dataset

from dingo.gw.domains import build_domain


class SVDBasis:
    def __init__(self):
        self.V = None
        self.Vh = None
        self.n = None

    def generate_basis(self, training_data: np.ndarray, n: int,
                       method: str = 'random'):
        """Generate the SVD basis from training data and store it.

        The SVD decomposition takes

        training_data = U @ diag(s) @ Vh

        where U and Vh are unitary.

        Parameters
        ----------
        training_data: np.ndarray
            Array of waveform data on the physical domain

        n: int
            Number of basis elements to keep.
            n=0 keeps all basis elements.
        method: str
            Select SVD method, 'random' or 'scipy'
        """
        if method == 'random':
            if n == 0:
                n = min(training_data.shape)

            U, s, Vh = randomized_svd(training_data, n)

            self.Vh = Vh.astype(np.complex64)
            self.V = self.Vh.T.conj()
            self.n = n
        elif method == 'scipy':
            # Code below uses scipy's svd tool. Likely slower.
            U, s, Vh = scipy.linalg.svd(training_data, full_matrices=False)
            V = Vh.T.conj()

            if (n == 0) or (n > len(V)):
                self.V = V
                self.Vh = Vh
            else:
                self.V = V[:, :n]
                self.Vh = Vh[:n, :]

            self.n = len(self.Vh)
        else:
            raise ValueError(f'Unsupported SVD method: {method}.')

    def basis_coefficients_to_fseries(self, coefficients: np.ndarray):
        """
        Convert from basis coefficients to frequency series.

        Parameters
        ----------
        coefficients:
            Array of basis coefficients
        """
        return coefficients @ self.Vh

    def fseries_to_basis_coefficients(self, fseries: np.ndarray):
        """
        Convert from frequency series to basis coefficients.

        Parameters
        ----------
        fseries:
            Array of frequency series
        """
        return fseries @ self.V

    def from_file(self, filename: str):
        """
        Load basis matrix V from a file.

        Parameters
        ----------
        filename:
            File in .npy format
        """
        self.V = np.load(filename)
        self.Vh = self.V.T.conj()
        self.n = self.V.shape[1]

    def to_file(self, filename: str):
        """
        Save basis matrix V to a file.

        Parameters
        ----------
        filename:
            File in .npy format
        """
        if self.V is not None:
            np.save(filename, self.V)


class WaveformDataset(Dataset):
    """This class loads a dataset of simulated waveforms (plus and cross
    polarizations, as well as associated parameter values.

    This class loads a stored set of waveforms from an HDF5 file.
    Waveform polarizations are generated by the scripts in
    gw.waveform_dataset_generation.
    Once a waveform data set is in memory, the waveform data are consumed through
    a __getitem__() call, optionally applying a chain of transformations, which
    are classes that implement the __call__() method.
    """

    def __init__(self, dataset_file: str, transform=None):
        """
        Parameters
        ----------
        dataset_file : str
            Load the waveform dataset from this HDF5 file.
        transform :
            Transformations to apply.
        """
        self.transform = transform
        self._Vh = None
        self.load(dataset_file)


    def __len__(self):
        """The number of waveform samples."""
        return len(self._parameter_samples)


    def __getitem__(self, idx) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Return a nested dictionary containing parameters and waveform polarizations
        for sample with index `idx`. If defined a chain of transformations are being
        applied to the waveform data.
        """
        parameters = self._parameter_samples.iloc[idx].to_dict()
        waveform_polarizations = self._waveform_polarizations.iloc[idx].to_dict()
        data = {'parameters': parameters, 'waveform': waveform_polarizations}
        if '_Vh' in self.__dict__:
            data['waveform']['h_plus'] = data['waveform']['h_plus'] @ self._Vh
            data['waveform']['h_cross'] = data['waveform']['h_cross'] @ self._Vh
        if self.transform:
            data = self.transform(data)
        return data


    def get_info(self):
        """
        Print information on the stored pandas DataFrames.
        This is before any transformations are done.
        """
        self._parameter_samples.info(memory_usage='deep')
        self._waveform_polarizations.info(memory_usage='deep')


    def load(self, filename: str = 'waveform_dataset.h5'):
        """
        Load waveform data set from HDF5 file.

        Parameters
        ----------
        filename : str
            The name of the HDF5 file containing the data set.
        """
        fp = h5py.File(filename, 'r')

        parameter_array = fp['parameters'][:]
        self._parameter_samples = pd.DataFrame(parameter_array)

        grp = fp['waveform_polarizations']
        polarization_dict_2d = {k: v[:] for k, v in grp.items()}
        polarization_dict = {k: [x for x in polarization_dict_2d[k]]
                             for k in ['h_plus', 'h_cross']}
        self._waveform_polarizations = pd.DataFrame(polarization_dict)

        if 'rb_matrix_V' in fp.keys():
            V = fp['rb_matrix_V'][:]
            self._Vh = V.T.conj()

        self.data_settings = ast.literal_eval(fp.attrs['settings'])
        self.domain = build_domain(self.data_settings['domain_settings'])

        fp.close()
