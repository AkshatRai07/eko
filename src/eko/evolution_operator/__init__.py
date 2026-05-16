r"""Contains the central operator classes.

See :doc:`Operator overview </code/Operators>`.
"""

import functools
import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Tuple

import numba as nb
import numpy as np

from scipy import LowLevelCallable, integrate

from .. import basis_rotation as br
from .. import interpolation, mellin
from .. import scale_variations as sv
from ..couplings import Couplings
from ..interpolation import InterpolatorDispatcher
from ..io.types import EvolutionMethod, OperatorLabel
from ..kernels import ev_method
from ..matchings import Atlas, Segment
from ..member import OpMember
from .quad_ker import cb_quad_ker_qcd, cb_quad_ker_qed

import ekors

logger = logging.getLogger(__name__)


spec = [
    ("is_singlet", nb.boolean),
    ("is_QEDsinglet", nb.boolean),
    ("is_QEDvalence", nb.boolean),
    ("is_log", nb.boolean),
    ("logx", nb.float64),
    ("u", nb.float64),
]


@nb.experimental.jitclass(spec)
class QuadKerBase:
    """Manage the common part of Mellin inversion integral.

    Parameters
    ----------
    u : float
        quad argument
    is_log : boolean
        is a logarithmic interpolation
    logx : float
        Mellin inversion point
    mode0 : str
        first sector element
    """

    def __init__(self, u, is_log, logx, mode0):
        self.is_singlet = mode0 in [100, 21, 90]
        self.is_QEDsinglet = mode0 in [21, 22, 100, 101, 90]
        self.is_QEDvalence = mode0 in [10200, 10204]
        self.is_log = is_log
        self.u = u
        self.logx = logx

    @property
    def path(self):
        """Return the associated instance of :class:`eko.mellin.Path`."""
        if self.is_singlet or self.is_QEDsinglet:
            return mellin.Path(self.u, self.logx, True)
        else:
            return mellin.Path(self.u, self.logx, False)

    @property
    def n(self):
        """Returs the Mellin moment :math:`N`."""
        return self.path.n

    def integrand(
        self,
        areas,
    ):
        """Get transformation to Mellin space integral.

        Parameters
        ----------
        areas : tuple
            basis function configuration

        Returns
        -------
        complex
            common mellin inversion integrand
        """
        if self.logx == 0.0:
            return 0.0
        pj = interpolation.evaluate_grid(self.path.n, self.is_log, self.logx, areas)
        if pj == 0.0:
            return 0.0
        return self.path.prefactor * pj * self.path.jac



OpMembers = Dict[OperatorLabel, OpMember]
"""Map of all operators."""


@dataclass(frozen=True)
class Managers:
    """Set of steering objects."""

    atlas: Atlas
    couplings: Couplings
    interpolator: InterpolatorDispatcher


class Operator(sv.ScaleVariationModeMixin):
    """Internal representation of a single EKO.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
    config : dict
        configuration
    managers : dict
        managers
    nf : int
        number of active flavors
    q2_from : float
        evolution source
    q2_to : float
        evolution target
    mellin_cut : float
        cut to the upper limit in the mellin inversion
    is_threshold : bool
        is this an itermediate threshold operator?
    """

    log_label = "Evolution"
    # complete list of possible evolution operators labels
    full_labels: Tuple[OperatorLabel, ...] = br.full_labels
    full_labels_qed: Tuple[OperatorLabel, ...] = br.full_unified_labels

    def __init__(
        self,
        config,
        managers: Managers,
        segment: Segment,
        mellin_cut=5e-2,
        is_threshold=False,
    ):
        self.config = config
        self.managers = managers
        self.nf = segment.nf
        self.q2_from = segment.origin
        self.q2_to = segment.target
        # TODO make 'cut' external parameter?
        self._mellin_cut = mellin_cut
        self.is_threshold = is_threshold
        self.op_members: OpMembers = {}
        self.order = tuple(config["order"])
        self.alphaem_running = self.managers.couplings.alphaem_running
        if self.log_label == "Evolution":
            self.a = self.compute_a()
            self.as_list, self.a_half_list = self.compute_aem_list()

    @property
    def n_pools(self):
        """Return number of parallel cores."""
        n_pools = self.config["n_integration_cores"]
        if n_pools > 0:
            return n_pools
        # so we subtract from the maximum number
        return max(os.cpu_count() + n_pools, 1)

    @property
    def xif2(self):
        r"""Return scale variation factor :math:`(\mu_F/\mu_R)^2`."""
        return self.config["xif2"]

    @property
    def int_disp(self) -> InterpolatorDispatcher:
        """Return the interpolation dispatcher."""
        return self.managers.interpolator

    @property
    def grid_size(self):
        """Return the grid size."""
        return self.int_disp.xgrid.size

    @property
    def mu2(self):
        """Return the arguments to the :math:`a_s` function."""
        mu0 = self.q2_from * (
            self.xif2 if self.sv_mode == sv.Modes.exponentiated else 1.0
        )
        mu1 = self.q2_to * (
            self.xif2
            if (not self.is_threshold and self.sv_mode == sv.Modes.expanded)
            or self.sv_mode == sv.Modes.exponentiated
            else 1.0
        )
        return (mu0, mu1)

    def compute_a(self):
        """Return the computed values for :math:`a_s` and :math:`a_{em}`."""
        coupling = self.managers.couplings
        a0 = coupling.a(
            self.mu2[0],
            nf_to=self.nf,
        )
        a1 = coupling.a(
            self.mu2[1],
            nf_to=self.nf,
        )
        return (a0, a1)

    @property
    def a_s(self):
        """Return the computed values for :math:`a_s`."""
        return (self.a[0][0], self.a[1][0])

    @property
    def a_em(self):
        """Return the computed values for :math:`a_{em}`."""
        return (self.a[0][1], self.a[1][1])

    def compute_aem_list(self):
        """Return the list of the couplings for the different values of
        :math:`a_s`.

        This functions is needed in order to compute the values of :math:`a_s`
        and :math:`a_em` in the middle point of the :math:`mu^2` interval, and
        the values of :math:`a_s` at the borders of every intervals.
        This is needed in the running_alphaem solution.
        """
        ev_op_iterations = self.config["ev_op_iterations"]
        if self.order[1] == 0:
            as_list = np.array([self.a_s[0], self.a_s[1]])
            a_half = np.zeros((ev_op_iterations, 2))
        else:
            couplings = self.managers.couplings
            mu2_steps = np.geomspace(self.q2_from, self.q2_to, 1 + ev_op_iterations)
            mu2_l = mu2_steps[0]
            as_list = np.array(
                [couplings.a_s(scale_to=mu2, nf_to=self.nf) for mu2 in mu2_steps]
            )
            a_half = np.zeros((ev_op_iterations, 2))
            for step, mu2_h in enumerate(mu2_steps[1:]):
                mu2_half = (mu2_h + mu2_l) / 2.0
                a_s, aem = couplings.a(scale_to=mu2_half, nf_to=self.nf)
                a_half[step] = [a_s, aem]
                mu2_l = mu2_h
        return as_list, a_half

    @property
    def labels(self):
        """Compute necessary sector labels to compute.

        Returns
        -------
        list(str)
            sector labels
        """
        labels = []
        # the NS sector is dynamic
        if self.order[1] == 0:
            if self.config["debug_skip_non_singlet"]:
                logger.warning("%s: skipping non-singlet sector", self.log_label)
            else:
                # add + as default
                labels.append(br.non_singlet_labels[1])
                if self.order[0] >= 2:  # - becomes different starting from NLO
                    labels.append(br.non_singlet_labels[0])
                if self.order[0] >= 3:  # v also becomes different starting from NNLO
                    labels.append(br.non_singlet_labels[2])
            # singlet sector is fixed
            if self.config["debug_skip_singlet"]:
                logger.warning("%s: skipping singlet sector", self.log_label)
            else:
                labels.extend(br.singlet_labels)
        else:
            if self.config["debug_skip_non_singlet"]:
                logger.warning("%s: skipping non-singlet sector", self.log_label)
            else:
                # add +u and +d as default
                labels.append(br.non_singlet_unified_labels[0])
                labels.append(br.non_singlet_unified_labels[2])
                # -u and -d become different starting from O(as1aem1) or O(aem2)
                # but at this point order is at least (1, 1)
                labels.append(br.non_singlet_unified_labels[1])
                labels.append(br.non_singlet_unified_labels[3])
                labels.extend(br.valence_unified_labels)
            if self.config["debug_skip_singlet"]:
                logger.warning("%s: skipping singlet sector", self.log_label)
            else:
                labels.extend(br.singlet_unified_labels)
        return labels

    @property
    def ev_method(self):
        """Return the evolution method."""
        return ev_method(EvolutionMethod(self.config["method"]))

    def initialize_op_members(self):
        """Init all operators with the identity or zeros."""
        eye = OpMember(
            np.eye(self.grid_size), np.zeros((self.grid_size, self.grid_size))
        )
        zero = OpMember(*[np.zeros((self.grid_size, self.grid_size))] * 2)
        if self.order[1] == 0:
            full_labels = self.full_labels
            non_singlet_labels = br.non_singlet_labels
        else:
            full_labels = self.full_labels_qed
            non_singlet_labels = br.non_singlet_unified_labels
        for n in full_labels:
            if n in self.labels:
                # non-singlet evolution and diagonal op are identities
                if n in non_singlet_labels or n[0] == n[1]:
                    self.op_members[n] = eye.copy()
                else:
                    self.op_members[n] = zero.copy()
            else:
                self.op_members[n] = zero.copy()

    def update_cfg(self, cfg):
        """Adjust integration config."""
        # prepare couplings for c
        cfg.as1 = self.as_list[1]
        cfg.as0 = self.as_list[0]
        as_list_len = self.as_list.shape[0]
        for j, a in enumerate(self.as_list.tolist()):
            cfg.as_list[j] = a
        cfg.as_list_len = as_list_len
        a_half_x = self.a_half_list.shape[0]
        a_half_y = self.a_half_list.shape[1]
        for j, a in enumerate(self.a_half_list.flatten().tolist()):
            cfg.a_half[j] = a
        cfg.a_half_x = a_half_x
        cfg.a_half_y = a_half_y

        if self.order[1] == 0:
            cfg.py = ekors.ffi.cast("void *", cb_quad_ker_qcd.address)
        else:
            cfg.py = ekors.ffi.cast("void *", cb_quad_ker_qed.address)
        cfg.method_num = self.ev_method

    def run_op_integration(self, log_grid):
        """Run the integration for each grid point.

        Parameters
        ----------
        log_grid : tuple(k, logx)
            log grid point with relative index

        Returns
        -------
        list
            computed operators at the give grid point
        """
        column = []
        k, logx = log_grid
        # call(!) self.labels only once
        labels = self.labels
        start_time = time.perf_counter()
        # start preparing C arguments
        cfg = ekors.lib.empty_args()
        cfg.order_qcd = self.order[0]
        cfg.order_qed = self.order[1]
        cfg.is_polarized = self.config["polarized"]
        cfg.is_time_like = self.config["time_like"]
        cfg.nf = self.nf
        cfg.is_log = self.int_disp.log
        cfg.logx = logx
        cfg.Lsv = np.log(self.xif2)
        cfg.ev_op_iterations = self.config["ev_op_iterations"]
        cfg.ev_op_max_order_qcd = self.config["ev_op_max_order"][0]
        cfg.sv_mode_num = self.sv_mode
        cfg.is_threshold = self.is_threshold
        cfg.mu2_from = self.q2_from
        cfg.mu2_to = self.q2_to
        cfg.alphaem_running = self.alphaem_running
        a_half_ffi = ekors.ffi.new("double[]", 2 * self.config["ev_op_iterations"])
        cfg.a_half = a_half_ffi
        as_list_ffi = ekors.ffi.new("double[]", self.config["ev_op_iterations"] + 1)
        cfg.as_list = as_list_ffi
        max_areas_shape = self.int_disp.max_areas_shape()
        max_area_len = max_areas_shape[0] * max_areas_shape[1]
        areas_ffi = ekors.ffi.new("double[]", max_area_len)
        cfg.areas = areas_ffi

        self.update_cfg(cfg)

        # iterate basis functions
        for j, bf in enumerate(self.int_disp):
            if k == j and j == self.grid_size - 1:
                continue
            # add emtpy labels with 0s
            if bf.is_below_x(np.exp(logx)):
                column.append({label: (0.0, 0.0) for label in labels})
                continue
            temp_dict = {}
            # prepare areas for C
            curareas = bf.areas_representation
            for j, x in enumerate(curareas.flatten().tolist()):
                cfg.areas[j] = x
            cfg.areas_x = curareas.shape[0]
            cfg.areas_y = curareas.shape[1]
            # iterate sectors
            for label in labels:
                cfg.mode0 = label[0]
                cfg.mode1 = label[1]
                # construct the low level object
                func = LowLevelCallable(
                    ekors.lib.rust_quad_ker, ekors.ffi.addressof(cfg)
                )
                res = integrate.quad(
                    func,
                    0.5,
                    1.0 - self._mellin_cut,
                    epsabs=1e-12,
                    epsrel=1e-5,
                    limit=100,
                    full_output=1,
                )
                temp_dict[label] = res[:2]
            column.append(temp_dict)
        logger.info(
            "%s: computing operators - %u/%u took: %6f s",
            self.log_label,
            k + 1,
            self.grid_size,
            (time.perf_counter() - start_time),
        )
        return column

    def compute(self):
        """Compute the actual operators (i.e. run the integrations)."""
        self.initialize_op_members()

        # skip computation?
        if np.isclose(self.q2_from, self.q2_to):
            # unless we have to do some scale variation
            # TODO remove if K is factored out of here
            if not (
                self.sv_mode == sv.Modes.expanded
                and not np.isclose(self.xif2, 1.0)
                and not self.is_threshold
            ):
                logger.info(
                    "%s: skipping unity operator at %e", self.log_label, self.q2_from
                )
                self.copy_ns_ops()
                return

        logger.info(
            "%s: computing operators %e -> %e, nf=%d",
            self.log_label,
            self.q2_from,
            self.q2_to,
            self.nf,
        )
        logger.info(
            "%s: µ_R^2 distance: %e -> %e",
            self.log_label,
            self.mu2[0],
            self.mu2[1],
        )
        if self.sv_mode != sv.Modes.unvaried:
            logger.info(
                "Scale Variation: (µ_F/µ_R)^2 = %e, mode: %s",
                self.xif2,
                self.sv_mode.name,
            )
        logger.info(
            "%s: a_s distance: %e -> %e", self.log_label, self.a_s[0], self.a_s[1]
        )
        if self.order[1] > 0:
            logger.info(
                "%s: a_em distance: %e -> %e",
                self.log_label,
                self.a_em[0],
                self.a_em[1],
            )
        logger.info(
            "%s: order: (%d, %d), solution strategy: %s, use fhmruvv: %s",
            self.log_label,
            self.order[0],
            self.order[1],
            self.config["method"],
            self.config["use_fhmruvv"],
        )

        self.integrate()
        # copy non-singlet kernels, if necessary
        self.copy_ns_ops()

    def integrate(
        self,
    ):
        """Run the integration."""
        tot_start_time = time.perf_counter()

        # run integration in parallel for each grid point
        # or avoid opening a single pool
        args = (self.run_op_integration, enumerate(np.log(self.int_disp.xgrid.raw)))
        if self.n_pools == 1:
            res = map(*args)
        else:
            with Pool(self.n_pools) as pool:
                res = pool.map(*args)

        # collect results
        for j, row in enumerate(res):
            for k, entry in enumerate(row):
                for label, (val, err) in entry.items():
                    self.op_members[label].value[j][k] = val
                    self.op_members[label].error[j][k] = err

        # closing comment
        logger.info(
            "%s: Total time %f s",
            self.log_label,
            time.perf_counter() - tot_start_time,
        )

    def copy_ns_ops(self):
        """Copy non-singlet kernels, if necessary."""
        if self.order[1] == 0:
            if self.order[0] == 1:  # in LO +=-=v
                for label in ["nsV", "ns-"]:
                    self.op_members[
                        (br.non_singlet_pids_map[label], 0)
                    ].value = self.op_members[
                        (br.non_singlet_pids_map["ns+"], 0)
                    ].value.copy()
                    self.op_members[
                        (br.non_singlet_pids_map[label], 0)
                    ].error = self.op_members[
                        (br.non_singlet_pids_map["ns+"], 0)
                    ].error.copy()
            elif self.order[0] == 2:  # in NLO -=v
                self.op_members[
                    (br.non_singlet_pids_map["nsV"], 0)
                ].value = self.op_members[
                    (br.non_singlet_pids_map["ns-"], 0)
                ].value.copy()
                self.op_members[
                    (br.non_singlet_pids_map["nsV"], 0)
                ].error = self.op_members[
                    (br.non_singlet_pids_map["ns-"], 0)
                ].error.copy()
        # at O(as0aem1) u-=u+, d-=d+
        # starting from O(as1aem1) P+ != P-
        # However the solution with pure QED is not implemented in EKO
        # so the ns anomalous dimensions are always different
        # elif self.order[1] == 1 and self.order[0] == 0:
        #     self.op_members[
        #         (br.non_singlet_pids_map["ns-u"], 0)
        #     ].value = self.op_members[(br.non_singlet_pids_map["ns+u"], 0)].value.copy()
        #     self.op_members[
        #         (br.non_singlet_pids_map["ns-u"], 0)
        #     ].error = self.op_members[(br.non_singlet_pids_map["ns+u"], 0)].error.copy()
        #     self.op_members[
        #         (br.non_singlet_pids_map["ns-d"], 0)
        #     ].value = self.op_members[(br.non_singlet_pids_map["ns+d"], 0)].value.copy()
        #     self.op_members[
        #         (br.non_singlet_pids_map["ns-d"], 0)
        #     ].error = self.op_members[(br.non_singlet_pids_map["ns+d"], 0)].error.copy()
