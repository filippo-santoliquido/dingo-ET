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
