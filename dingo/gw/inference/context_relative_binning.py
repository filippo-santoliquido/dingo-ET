"""Sampler context for relative-binning summary data (RelativeBinningDomain, Stage B).

The event data are the unwhitened summary vectors A per detector (2B complex: A_0 of all
bins, then A_1), computed for the reference of fixed proxies, which the event metadata
carry as `relative_binning_proxies` = {"chirp_mass_proxy": ..., "t_star_proxy": ...}. The
network input is the per-bin whitened x = L^-1 A; the likelihood is the relative-binning
likelihood on the same summaries. A change of proxies (recentre) needs new summaries, so a
new context.
"""

from typing import Optional, Union

import numpy as np
import torch
from torchvision.transforms import Compose

from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_interferometer_list
from dingo.gw.inference.context import GWSamplerContext, _event_detectors
from dingo.gw.likelihood_relative_binning import RelativeBinningLikelihood
from dingo.core.transforms import GetItem
from dingo.gw.transforms import RepackageStrainsAndASDS, ToTorch
from dingo.gw.transforms.relative_binning_transforms import WhitenSummary


class RelativeBinningSamplerContext(GWSamplerContext):
    """GWSamplerContext for summary data at fixed proxies (see module docstring)."""

    @classmethod
    def from_model_metadata(
        cls,
        metadata: dict,
        event_data: dict,
        event_metadata: Optional[dict] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> "RelativeBinningSamplerContext":
        if cls is RelativeBinningSamplerContext and metadata["dataset_settings"].get(
            "polarizations_per_m", False
        ):
            # Per-m-group summaries (Stage D): their own data input and likelihood.
            return ModeGroupSamplerContext.from_model_metadata(
                metadata, event_data, event_metadata, device
            )
        domain = build_domain(metadata["dataset_settings"]["domain"])
        detectors = _event_detectors(metadata["train_settings"]["data"], event_metadata)
        ifo_names = [ifo.name for ifo in get_interferometer_list(detectors)]
        if event_metadata is None or "relative_binning_proxies" not in event_metadata:
            raise ValueError(
                "Summary data belong to fixed proxies: event_metadata needs "
                "relative_binning_proxies (chirp_mass_proxy, t_star_proxy)."
            )
        transforms = [
            WhitenSummary({n: domain.gram[n] for n in ifo_names}),
            RepackageStrainsAndASDS(ifos=ifo_names, first_index=0),
            ToTorch(device=device),
            GetItem("waveform"),
        ]
        return cls(
            domain=domain,
            data_prep=Compose(transforms),
            event_data=event_data,
            event_metadata=event_metadata,
            model_metadata=metadata,
            device=device,
        )

    def _event_grid(self):
        return self.domain

    def _validate_event_settings(self):
        if (self.event_metadata or {}).get("psd_notch_dict") or any(
            k in (self.event_metadata or {})
            for k in ("minimum_frequency", "maximum_frequency")
        ):
            raise NotImplementedError(
                "Frequency-range updates and PSD notches are not supported for "
                "relative-binning summary data."
            )
        super()._validate_event_settings()

    def prepared_data(self, conditioning=None):
        """As GWSamplerContext.prepared_data; the pinned proxies must be those of the
        summaries."""
        proxies = self.event_metadata["relative_binning_proxies"]
        for name, value in (conditioning or {}).items():
            if name in proxies:
                v = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
                # the chain may hold the pins in float32 (relative precision 6e-8)
                if not torch.allclose(v, torch.full_like(v, proxies[name]), rtol=1e-6, atol=1e-6):
                    raise ValueError(
                        f"{name} = {v[0].item()} differs from the proxy of the summary "
                        f"data ({proxies[name]}); compute summaries for this proxy."
                    )
        return super().prepared_data(conditioning)

    def likelihood(
        self,
        time_marginalization_kwargs: Optional[dict] = None,
        phase_marginalization_kwargs: Optional[dict] = None,
        calibration_marginalization_kwargs: Optional[dict] = None,
        use_base_domain: bool = False,
        wfg_updates: Optional[dict] = None,
        heterodyning_parameters: Optional[dict] = None,
    ) -> RelativeBinningLikelihood:
        """The relative-binning likelihood on the event summaries (no marginalizations)."""
        if time_marginalization_kwargs or phase_marginalization_kwargs or (
            calibration_marginalization_kwargs
        ):
            raise NotImplementedError(
                "Marginalizations are not implemented for the relative-binning likelihood."
            )
        if self._likelihood is None:
            metadata = self.event_metadata
            wfg = dict(self.model_metadata["dataset_settings"]["waveform_generator"])
            wfg.update(wfg_updates or {})
            self._likelihood = RelativeBinningLikelihood(
                self.domain,
                wfg,
                self.event_data["waveform"],
                metadata["relative_binning_proxies"],
                self.detectors,
                metadata.get("time_event", self.t_ref),
                log_Zn=metadata.get("log_noise_evidence", 0.0),
                # the event's response (e.g. to test a network on data with another
                # response), else the one of the training
                detector_response=metadata.get(
                    "detector_response",
                    self.model_metadata["train_settings"]["data"].get("detector_response"),
                ),
            )
        return self._likelihood


class ModeGroupSamplerContext(RelativeBinningSamplerContext):
    """Sampler context of a per-m-group network (Stage D; `dingo.gw.relative_binning_mode_data`).

    event_data:
        "waveform": {detector: (G, 2, B) complex} network summaries A (tapered references)
            of one data pass at the proxies;
        "mode_group_summaries": {"D": {detector: (G, 2, B)}, "gram": {detector:
            (G, G, 3, B)}, "gram_tapered": {detector: (G, G, 3, B)}} from the same pass.
    event_metadata: "relative_binning_proxies" (chirp_mass_proxy on the lookup grid,
    t_star_proxy), "log_noise_evidence", "time_event"; optional "mode_group_taper"
    ("reference" or "own", the likelihood's taper model) and "top_up_noise_seed".

    The base context keeps only the per-detector parts "waveform" and "asds" of
    event_data (and broadcasts them over batch rows), so "mode_group_summaries" is taken
    out first and kept in `self.mode_group_summaries` for the likelihood. It is not part
    of the stored Result context: a context rebuilt from a saved Result needs it again
    (the injection script saves it in summaries.hdf5).
    """

    def __init__(self, *args, event_data=None, **kwargs):
        if event_data is not None:
            event_data = dict(event_data)
            self.mode_group_summaries = event_data.pop("mode_group_summaries", None)
        else:
            self.mode_group_summaries = None
        super().__init__(*args, event_data=event_data, **kwargs)

    @classmethod
    def from_model_metadata(cls, metadata, event_data, event_metadata=None, device="cpu"):
        from dingo.gw.relative_binning_mode_tables import round_to_grid, tables_for_dataset
        from dingo.gw.transforms.relative_binning_mode_transforms import (
            WhitenModeGroupSummary,
        )

        domain = build_domain(metadata["dataset_settings"]["domain"])
        data_settings = metadata["train_settings"]["data"]
        detectors = _event_detectors(data_settings, event_metadata)
        ifos = get_interferometer_list(detectors)
        if event_metadata is None or "relative_binning_proxies" not in event_metadata:
            raise ValueError(
                "Summary data belong to fixed proxies: event_metadata needs "
                "relative_binning_proxies (chirp_mass_proxy, t_star_proxy)."
            )
        lookup_meta = _lookup_meta(data_settings.get("mode_group_lookup"))
        mc = float(event_metadata["relative_binning_proxies"]["chirp_mass_proxy"])
        if abs(float(round_to_grid(mc, lookup_meta["delta"])) - mc) > 1e-9 * lookup_meta["delta"]:
            raise ValueError(
                f"The chirp-mass proxy {mc} is not on the lookup grid "
                f"(delta {lookup_meta['delta']}): the network was trained on grid proxies."
            )
        wg = metadata["dataset_settings"]["waveform_generator"]
        tables = {
            ifo.name: tables_for_dataset(
                domain, wg, ifo, eps=lookup_meta["eps"], n_nodes=lookup_meta["n_nodes"]
            )
            for ifo in ifos
        }
        spec = data_settings.get("mode_group_lookup")
        for name, tab in tables.items():
            meta = _lookup_meta(spec.get(name) if isinstance(spec, dict) else spec)
            if "weight_sha1" in meta and tab.settings()["weight_sha1"] != meta["weight_sha1"]:
                raise ValueError(
                    f"The PSD of {name} differs from the PSD of the network's lookup table."
                )
        transforms = [
            WhitenModeGroupSummary(
                tables, mc, seed=(event_metadata or {}).get("top_up_noise_seed", 0)
            ),
            RepackageStrainsAndASDS(ifos=[ifo.name for ifo in ifos], first_index=0),
            ToTorch(device=device),
            GetItem("waveform"),
        ]
        return cls(
            domain=domain,
            data_prep=Compose(transforms),
            event_data=event_data,
            event_metadata=event_metadata,
            model_metadata=metadata,
            device=device,
        )

    def likelihood(
        self,
        time_marginalization_kwargs: Optional[dict] = None,
        phase_marginalization_kwargs: Optional[dict] = None,
        calibration_marginalization_kwargs: Optional[dict] = None,
        use_base_domain: bool = False,
        wfg_updates: Optional[dict] = None,
        heterodyning_parameters: Optional[dict] = None,
    ):
        """The per-m-group relative-binning likelihood on the event summaries."""
        from dingo.gw.likelihood_relative_binning import ModeGroupRelativeBinningLikelihood

        if time_marginalization_kwargs or phase_marginalization_kwargs or (
            calibration_marginalization_kwargs
        ):
            raise NotImplementedError(
                "Marginalizations are not implemented for the relative-binning likelihood."
            )
        if self._likelihood is None:
            metadata = self.event_metadata
            wfg = dict(self.model_metadata["dataset_settings"]["waveform_generator"])
            wfg.update(wfg_updates or {})
            taper = metadata.get("mode_group_taper", "reference")
            extra = self.mode_group_summaries
            if extra is None:
                raise ValueError(
                    "No mode_group_summaries (D, gram, gram_tapered) for the per-m "
                    "likelihood: pass them in event_data (they are not stored in a "
                    "saved Result)."
                )
            if taper == "reference":
                summary, gram = self.event_data["waveform"], extra["gram_tapered"]
            else:
                summary, gram = extra["D"], extra["gram"]
            self._likelihood = ModeGroupRelativeBinningLikelihood(
                self.domain,
                wfg,
                summary,
                gram,
                metadata["relative_binning_proxies"],
                self.detectors,
                metadata.get("time_event", self.t_ref),
                log_Zn=metadata.get("log_noise_evidence", 0.0),
                detector_response=metadata.get(
                    "detector_response",
                    self.model_metadata["train_settings"]["data"].get("detector_response"),
                ),
                taper=taper,
            )
        return self._likelihood


def _lookup_meta(path):
    """meta.json of the network's lookup table (eps, nodes, grid, PSD weights); the
    defaults of training (eps 1e-2, 8 nodes, grid 1e-4 Msun) if it is not on this
    machine."""
    import json
    import os
    import warnings

    if isinstance(path, dict):  # one table per detector: same eps, nodes and grid
        path = next(iter(path.values()), None)
    if path is not None and os.path.exists(os.path.join(str(path), "meta.json")):
        with open(os.path.join(str(path), "meta.json")) as f:
            return json.load(f)
    warnings.warn(
        f"Lookup table {path} not found: eps 1e-2, 8 nodes, grid 1e-4 Msun assumed; the "
        "PSD is not checked against the training."
    )
    return {"eps": 1e-2, "n_nodes": 8, "delta": 1e-4}
