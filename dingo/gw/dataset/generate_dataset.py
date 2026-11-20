import argparse
import copy
import json
import textwrap
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple
from functools import partial

import numpy as np
import pandas as pd
import yaml
from bilby.gw.prior import BBHPriorDict
from threadpoolctl import threadpool_limits
from torchvision.transforms import Compose

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import RelativeBinningDomain, build_domain
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.SVD import ApplySVD, SVDBasis
from dingo.gw.transforms import HeterodynePhase, WhitenFixedASD
from dingo.gw.transforms.relative_binning_transforms import (
    DivideByFiducial,
    FiducialPhase,
)
from dingo.gw.dataset.relative_binning_bins import complete_relative_binning_domain
from dingo.gw.waveform_generator import (
    NewInterfaceWaveformGenerator,
    WaveformGenerator,
    generate_waveforms_parallel,
)
from dingo.core.utils.misc import call_func_strict_output_dim


def generate_parameters_and_polarizations(
    waveform_generator: WaveformGenerator,
    prior: BBHPriorDict,
    num_samples: int,
    num_processes: int,
    per_m: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Generate a dataset of waveforms based on parameters drawn from the prior.

    Parameters
    ----------
    waveform_generator : WaveformGenerator
    prior : Prior
    num_samples : int
    num_processes : int
    per_m : bool
        Polarizations per m-component, keys f"{pol}_m{m}" (generate_waveforms_parallel).

    Returns
    -------
    pandas DataFrame of parameters
    dictionary of numpy arrays corresponding to waveform polarizations
    """
    print("Generating dataset of size " + str(num_samples))
    parameters = pd.DataFrame(prior.sample(num_samples))

    if num_processes > 1:
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                polarizations = generate_waveforms_parallel(
                    waveform_generator, parameters, pool, per_m=per_m
                )
    else:
        polarizations = generate_waveforms_parallel(
            waveform_generator, parameters, per_m=per_m
        )

    # Find cases where waveform generation failed and only return data for successful ones
    wf_failed = np.any([np.isnan(v).any(axis=1) for v in polarizations.values()], axis=0)
    if wf_failed.any():
        idx_failed = np.where(wf_failed)[0]
        idx_ok = np.where(~wf_failed)[0]
        polarizations_ok = {k: v[idx_ok] for k, v in polarizations.items()}
        parameters_ok = parameters.iloc[idx_ok]
        failed_percent = 100 * len(idx_failed) / len(parameters)
        print(
            f"{len(idx_failed)} out of {len(parameters)} configuration ({failed_percent:.1f}%) failed to generate."
        )
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(parameters.iloc[idx_failed])
        print(
            f"Only returning the {len(idx_ok)} successfully generated configurations."
        )
        return parameters_ok, polarizations_ok

    return parameters, polarizations


def train_svd_basis(dataset: WaveformDataset, size: int, n_train: int):
    """
    Train (and optionally validate) an SVD basis.

    Parameters
    ----------
    dataset : WaveformDataset
        Contains waveforms to be used for building SVD.
    size : int
        Number of elements to keep for the SVD basis.
    n_train : int
        Number of training waveforms to use. Remaining are used for validation. Note
        that the actual number of training waveforms is n_train * len(polarizations),
        since there is one waveform used for each polarization.

    Returns
    -------
    SVDBasis, n_train, n_test
        Since EOB waveforms can fail to generate, provide also the number used in
        training and validation.
    """
    # Prepare data for training and validation.
    train_data = np.vstack([val[:n_train] for val in dataset.polarizations.values()])
    test_data = np.vstack([val[n_train:] for val in dataset.polarizations.values()])
    test_parameters = pd.concat(
        [
            # I would like to save the polarization, but saving the dataframe with
            # string columns causes problems. Fix this later.
            # dataset.parameters.iloc[n_train:].assign(polarization=pol)
            dataset.parameters.iloc[n_train:]
            for pol in dataset.polarizations
        ]
    )
    test_parameters.reset_index(drop=True, inplace=True)

    print("Building SVD basis.")
    basis = SVDBasis()
    basis.generate_basis(train_data, size)

    assert np.allclose(basis.V[: dataset.domain.min_idx], 0)

    # Since there is a possibility that the size of the dataset returned by
    # generate_parameters_and_polarizations is smaller than requested, we don't assume
    # that there are n_test samples. Instead we just look at the size of the test
    # dataset.
    if test_data.size != 0:
        basis.compute_test_mismatches(
            test_data, parameters=test_parameters, verbose=True
        )

    # Return also the true number of samples. Some EOB waveforms may have failed to
    # generate, so this could be smaller than the number requested.
    n_ifos = len(dataset.polarizations)
    n_train = len(train_data) // n_ifos
    n_test = len(test_data) // n_ifos

    return basis, n_train, n_test


def train_group_svd_bases(dataset: WaveformDataset, size: int, n_train: int):
    """One SVD basis per m group (keys f"{pol}_m{m}"; both polarizations of a group train
    its basis). Returns ({f"m{m}": SVDBasis}, n_train, n_test)."""
    from dingo.gw.SVD import svd_group_key

    groups = sorted({svd_group_key(k) for k in dataset.polarizations})
    bases = {}
    for g in groups:
        keys = [k for k in dataset.polarizations if svd_group_key(k) == g]
        train_data = np.vstack([dataset.polarizations[k][:n_train] for k in keys])
        test_data = np.vstack([dataset.polarizations[k][n_train:] for k in keys])
        test_parameters = pd.concat([dataset.parameters.iloc[n_train:] for _ in keys])
        test_parameters.reset_index(drop=True, inplace=True)
        print(f"Building SVD basis of group {g}.")
        basis = SVDBasis()
        basis.generate_basis(train_data, size)
        if test_data.size != 0:
            basis.compute_test_mismatches(test_data, parameters=test_parameters, verbose=True)
        bases[g] = basis
    n_tr = min(n_train, len(dataset.parameters))
    return bases, n_tr, len(dataset.parameters) - n_tr


def load_group_svd_bases(file_name: str) -> dict:
    """{f"m{m}": SVDBasis} from the "svd" entry of a dataset file with per-group SVD."""
    import h5py

    from dingo.core.dataset import recursive_hdf5_load
    from dingo.gw.SVD import is_group_svd

    with h5py.File(file_name, "r") as f:
        d = recursive_hdf5_load(f["svd"])
    if not is_group_svd(d):
        raise ValueError(f"{file_name} has no per-group SVD bases.")
    return {g: SVDBasis(dictionary=v) for g, v in d.items()}


def generate_dataset(settings: Dict, num_processes: int) -> WaveformDataset:
    """
    Generate a waveform dataset.

    Parameters
    ----------
    settings : dict
        Dictionary of settings to configure the dataset
    num_processes : int

    Returns
    -------
    A WaveformDataset based on the settings.
    """

    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    if (
        settings["domain"]["type"] == "RelativeBinningDomain"
        and "bin_edges" not in settings["domain"]
    ):
        # Bin edges and Gram matrices from the relative-binning hyperparameters
        # (bilby_xG greedy selection); stored in the settings, so in the dataset file.
        print("Selecting relative-binning bins (bilby_xG).")
        settings["domain"], report = complete_relative_binning_domain(
            settings["domain"],
            settings["waveform_generator"],
            prior,
            ref_time=settings["domain"]["settings"].get("ref_time", 1.4e9),
            num_processes=num_processes,
        )
        settings["domain"]["settings"]["bin_selection_report"] = report
        print(f"...{report['n_bins']} bins.")
    domain = build_domain(settings["domain"])

    new_interface_flag = settings["waveform_generator"].get("new_interface", False)
    if new_interface_flag:
        waveform_generator = NewInterfaceWaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )
    else:
        waveform_generator = WaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )

    dataset_dict = {"settings": settings}

    # Each m group of modes stored separately (keys f"{pol}_m{m}"), mlgw_bns only.
    per_m = settings.get("polarizations_per_m", False)
    wg = settings["waveform_generator"]
    if per_m:
        if wg["approximant"] != "mlgw_bns":
            raise NotImplementedError("polarizations_per_m needs approximant mlgw_bns.")
        groups = [m for _, m in waveform_generator.mode_list]
        if len(set(groups)) != len(groups):
            raise ValueError(
                f"polarizations_per_m needs one mode per m (one fiducial phase per "
                f"group), got {waveform_generator.mode_list}."
            )

    compression_transforms = []
    if isinstance(domain, RelativeBinningDomain):
        # Always stored as h / exp(i Phi_fid(Mc)) at the bin edges (smooth; the training
        # conditioning multiplies the fiducial back in float64). Per m group, each
        # group divided by the phase of its own mode of the fiducial.
        fiducial = partial(
            FiducialPhase,
            frequencies=domain(),
            f_star=domain.settings.get("f_star", 100.0),
        )
        if per_m:
            fid = {
                m: fiducial(
                    WaveformGenerator(wg["approximant"], domain, wg["f_ref"]),
                    mode=(ell, m),
                )
                for ell, m in waveform_generator.mode_list
            }
        else:
            fid = fiducial(WaveformGenerator(wg["approximant"], domain, wg["f_ref"]))
        compression_transforms.append(DivideByFiducial(fid))
        waveform_generator.transform = Compose(compression_transforms)
    if "compression" in settings:

        if "whitening" in settings["compression"]:
            compression_transforms.append(
                WhitenFixedASD(
                    domain,
                    asd_file=settings["compression"]["whitening"],
                    inverse=False,
                )
            )

        if "phase_heterodyning" in settings["compression"]:
            # Heterodyne each waveform at its own chirp mass (DINGO-BNS). This must
            # precede decimation: the generator decimates approximants without a
            # frequency-domain implementation onto a multibanded domain before
            # applying the compression transforms.
            if waveform_generator._domain_transform is not None:
                raise NotImplementedError(
                    "phase_heterodyning requires the waveforms to be generated on "
                    "the dataset domain, but approximant "
                    f"{settings['waveform_generator']['approximant']} is generated "
                    "on the base domain and decimated afterwards."
                )
            compression_transforms.append(
                HeterodynePhase(
                    domain,
                    inverse=False,
                    **settings["compression"]["phase_heterodyning"],
                )
            )

        if "svd" in settings["compression"]:
            svd_settings = settings["compression"]["svd"]
            # One basis per m group (polarizations_per_m): {f"m{m}": SVDBasis}.
            per_group = svd_settings.get("per_group", False)
            if per_group and not per_m:
                raise ValueError("compression.svd.per_group needs polarizations_per_m.")

            # Load an SVD basis from file, if specified.
            if "file" in svd_settings:
                if per_group:
                    basis = load_group_svd_bases(svd_settings["file"])
                else:
                    basis = SVDBasis(file_name=svd_settings["file"])

            # Otherwise, generate the basis based on simulated waveforms.
            else:
                # If using whitened or heterodyned waveforms, then the SVD should be
                # based on these.
                waveform_generator.transform = Compose(compression_transforms)

                n_train = svd_settings["num_training_samples"]
                n_test = svd_settings.get("num_validation_samples", 0)

                func = partial(
                    generate_parameters_and_polarizations,
                    waveform_generator,
                    prior,
                    num_processes=num_processes,
                    per_m=per_m,
                )
                parameters, polarizations = call_func_strict_output_dim(
                    func, n_train + n_test
                )
                svd_dataset_settings = copy.deepcopy(settings)
                svd_dataset_settings["num_samples"] = len(parameters)
                del svd_dataset_settings["compression"]["svd"]

                # We build a WaveformDataset containing the SVD-training waveforms
                # because when constructed, it will automatically zero the waveforms
                # below f_min. This is useful for EOB waveforms, which are Fourier
                # transformed from time domain, and hence are nonzero below f_min. The
                # waveforms need to be zeroed below f_min because this corresponds to
                # setting the lower bound of the likelihood integral.

                svd_dataset = WaveformDataset(
                    dictionary={
                        "parameters": parameters,
                        "polarizations": polarizations,
                        "settings": svd_dataset_settings,
                    }
                )
                if per_group:
                    basis, n_train, n_test = train_group_svd_bases(
                        svd_dataset, svd_settings["size"], n_train
                    )
                else:
                    basis, n_train, n_test = train_svd_basis(
                        svd_dataset, svd_settings["size"], n_train
                    )

            if per_group:
                from dingo.gw.SVD import ApplyGroupSVD

                compression_transforms.append(ApplyGroupSVD(basis))
                dataset_dict["svd"] = {g: b.to_dictionary() for g, b in basis.items()}
            else:
                compression_transforms.append(ApplySVD(basis))
                dataset_dict["svd"] = basis.to_dictionary()

        waveform_generator.transform = Compose(compression_transforms)

    func = partial(
        generate_parameters_and_polarizations,
        waveform_generator,
        prior,
        num_processes=num_processes,
        per_m=per_m,
    )
    parameters, polarizations = call_func_strict_output_dim(
        func, settings["num_samples"]
    )
    dataset_dict["parameters"] = parameters
    dataset_dict["polarizations"] = polarizations

    dataset_dict[settings["num_samples"]] = len(parameters)
    dataset = WaveformDataset(dictionary=dataset_dict)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate a waveform dataset based on a settings file.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel waveform generation",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="waveform_dataset.hdf5",
        help="Name of file for storing dataset.",
    )
    return parser.parse_args()


def _generate_dataset_main(
    settings_file: str, out_file: str, num_processes: int
) -> None:
    if not Path(settings_file).is_file():
        raise FileNotFoundError(f"dataset generation, failed to find {settings_file}")
    if not Path(out_file).parent.is_dir():
        raise FileNotFoundError(
            f"dataset generation: can not create {out_file}: "
            f"{Path(out_file).parent} does not exist"
        )
    # Load settings
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    dataset = generate_dataset(settings, num_processes)
    dataset.to_file(str(out_file))
    if settings["domain"]["type"] == "RelativeBinningDomain":
        # The bin edges, the hyperparameters and the selection report, readable.
        d = settings["domain"]
        rb_file = Path(out_file).with_name(Path(out_file).stem + "_relative_binning.json")
        rb_file.write_text(
            json.dumps(
                {"bin_edges": d["bin_edges"], "settings": d["settings"]}, indent=1
            )
        )
        print(f"Wrote {rb_file}.")


def main() -> None:
    args = parse_args()
    _generate_dataset_main(args.settings_file, args.out_file, args.num_processes)


if __name__ == "__main__":
    main()
