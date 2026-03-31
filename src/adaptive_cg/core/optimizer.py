"""
Differentiable coarse-graining optimizer using PyTorch.

Learns a soft atom-to-bead assignment matrix via gradient descent,
then extracts a hard mapping at low temperature.  Only practical
for molecules with <= ~200 heavy atoms (N*B assignment matrix).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _require_torch():
    """Import torch or raise a clear error."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for differentiable CG optimization.\n"
            "Install it with:  pip install adaptive-cg[optimize]\n"
            "or:               pip install torch>=2.0"
        ) from None


class DifferentiableCGOptimizer:
    """Optimise atom-to-bead assignments via differentiable soft assignment.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Atom positions in nm.
    masses : np.ndarray, shape (N,)
        Atom masses in daltons.
    n_beads : int
        Target number of CG beads.
    epochs : int
        Number of optimisation steps.
    lr : float
        Initial learning rate for Adam.
    temp_start : float
        Starting softmax temperature (high = soft).
    temp_end : float
        Final softmax temperature (low = hard).
    """

    def __init__(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        n_beads: int,
        epochs: int = 2000,
        lr: float = 0.05,
        temp_start: float = 0.5,
        temp_end: float = 0.05,
    ) -> None:
        self.torch = _require_torch()

        self.positions_np = np.asarray(positions, dtype=np.float64)
        self.masses_np = np.asarray(masses, dtype=np.float64)
        self.n_atoms = self.positions_np.shape[0]
        self.n_beads = n_beads
        self.epochs = epochs
        self.lr = lr
        self.temp_start = temp_start
        self.temp_end = temp_end

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_logits(self):
        """Create logit matrix with sequential bias for convergence."""
        torch = self.torch
        N, B = self.n_atoms, self.n_beads

        logits = torch.zeros(N, B, dtype=torch.float64)

        # Sequential bias: atom i is biased toward bead floor(i * B / N).
        for i in range(N):
            target_bead = int(i * B / N)
            target_bead = min(target_bead, B - 1)
            logits[i, target_bead] = 2.0

        logits.requires_grad_(True)
        return logits

    def _get_temperature(self, epoch: int) -> float:
        """Cosine annealing from temp_start to temp_end."""
        progress = epoch / max(self.epochs - 1, 1)
        return self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (
            1 + math.cos(math.pi * progress)
        )

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _compute_loss(self, assignment, pos, masses):
        """Compute total loss from soft assignment matrix.

        Components
        ----------
        reconstruction : MSE between soft-CG pairwise distances and
                         atom-level mean pairwise distances.
        entropy        : Encourage peaked assignment (low entropy).
        bead_usage     : Penalise unused beads (mass sum near zero).
        ordering       : Encourage sequential atom-to-bead mapping.
        """
        torch = self.torch
        N, B = assignment.shape

        # --- Bead positions (mass-weighted) ---
        mass_col = masses.unsqueeze(1)                    # (N, 1)
        weighted_assign = assignment * mass_col            # (N, B)
        bead_mass = weighted_assign.sum(dim=0) + 1e-12    # (B,)
        # (B, 3) = (B, N) @ (N, 3), each row normalised by bead mass
        bead_pos = (weighted_assign.T @ pos) / bead_mass.unsqueeze(1)

        # --- CG pairwise distance matrix ---
        diff = bead_pos.unsqueeze(0) - bead_pos.unsqueeze(1)  # (B, B, 3)
        cg_dmat = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  # (B, B)

        # --- Reference pairwise distance matrix (atom-level) ---
        atom_diff = pos.unsqueeze(0) - pos.unsqueeze(1)       # (N, N, 3)
        atom_dmat = torch.sqrt((atom_diff ** 2).sum(dim=-1) + 1e-12)

        # Soft reference: weighted mean atom-atom distance per bead pair
        # ref(i,j) = sum_a sum_b  A(a,i) * A(b,j) * d(a,b) / (sum_a A(a,i) * sum_b A(b,j))
        # Efficient: (B, N) @ (N, N) @ (N, B)
        a_t = assignment.T                                     # (B, N)
        ref_dmat = (a_t @ atom_dmat @ assignment)              # (B, B)
        bead_count = assignment.sum(dim=0) + 1e-12             # (B,)
        ref_dmat = ref_dmat / (bead_count.unsqueeze(0) * bead_count.unsqueeze(1))

        # Upper triangle only
        triu_mask = torch.triu(torch.ones(B, B, dtype=torch.bool), diagonal=1)
        reconstruction = ((cg_dmat[triu_mask] - ref_dmat[triu_mask]) ** 2).mean()

        # --- Entropy regularisation (encourage peaked assignment) ---
        entropy = -(assignment * torch.log(assignment + 1e-12)).sum(dim=1).mean()

        # --- Bead usage penalty (encourage all beads used) ---
        bead_usage_frac = bead_mass / bead_mass.sum()
        target_frac = 1.0 / B
        bead_usage = ((bead_usage_frac - target_frac) ** 2).sum()

        # --- Sequential ordering penalty ---
        bead_indices = torch.arange(B, dtype=torch.float64)
        # Expected bead index for each atom
        expected_bead = (assignment * bead_indices.unsqueeze(0)).sum(dim=1)  # (N,)
        atom_order = torch.arange(N, dtype=torch.float64) * (B - 1) / max(N - 1, 1)
        ordering = ((expected_bead - atom_order) ** 2).mean()

        total = (
            reconstruction
            + 0.01 * entropy
            + 0.1 * bead_usage
            + 0.05 * ordering
        )
        return total, reconstruction.item()

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def optimize(self) -> dict[str, Any]:
        """Run differentiable CG optimisation.

        Returns
        -------
        dict with keys:
            mapping        : list[list[int]]  -- hard CG mapping
            rmse           : float             -- final RMSE from eval
            n_beads_used   : int               -- beads with at least one atom
            loss_history   : list[float]       -- loss per epoch
            final_temperature : float
        """
        torch = self.torch

        pos = torch.tensor(self.positions_np, dtype=torch.float64)
        masses = torch.tensor(self.masses_np, dtype=torch.float64)
        logits = self._init_logits()

        optimizer = torch.optim.Adam([logits], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01,
        )

        loss_history: list[float] = []
        final_temp = self.temp_start

        for epoch in range(self.epochs):
            temp = self._get_temperature(epoch)
            final_temp = temp

            # Soft assignment via temperature-scaled softmax
            assignment = torch.softmax(logits / temp, dim=1)

            loss, recon = self._compute_loss(assignment, pos, masses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())

            if epoch % 200 == 0 or epoch == self.epochs - 1:
                print(
                    f"  epoch {epoch:5d}/{self.epochs}  "
                    f"loss={loss.item():.6f}  recon={recon:.6f}  "
                    f"temp={temp:.4f}  lr={scheduler.get_last_lr()[0]:.6f}"
                )

        # ------------------------------------------------------------------
        # Extract hard mapping via argmax
        # ------------------------------------------------------------------
        with torch.no_grad():
            final_assignment = torch.softmax(logits / final_temp, dim=1)
            hard_assignment = final_assignment.argmax(dim=1).cpu().numpy()

        mapping: list[list[int]] = [[] for _ in range(self.n_beads)]
        for atom_idx, bead_idx in enumerate(hard_assignment):
            mapping[bead_idx].append(atom_idx)

        # Remove empty beads
        mapping = [group for group in mapping if len(group) > 0]
        n_beads_used = len(mapping)

        # ------------------------------------------------------------------
        # Final evaluation
        # ------------------------------------------------------------------
        rmse = self._eval_final(mapping)

        return {
            "mapping": mapping,
            "rmse": rmse,
            "n_beads_used": n_beads_used,
            "loss_history": loss_history,
            "final_temperature": final_temp,
        }

    def _eval_final(self, mapping: list[list[int]]) -> float:
        """Compute RMSE of the hard mapping, using eval_mapping if available."""
        try:
            from adaptive_cg.core.mapping import eval_mapping
            result = eval_mapping(mapping, self.positions_np, self.masses_np)
            return result["rmse"]
        except (ImportError, Exception):
            # Inline fallback: compute RMSE from bead-center pairwise distances
            return self._inline_rmse(mapping)

    def _inline_rmse(self, mapping: list[list[int]]) -> float:
        """Fallback RMSE computation without eval_mapping."""
        n_beads = len(mapping)
        if n_beads < 2:
            return 0.0

        positions = self.positions_np
        masses = self.masses_np

        # Bead centres of mass
        bead_pos = np.empty((n_beads, 3))
        for i, group in enumerate(mapping):
            grp = np.asarray(group)
            m = masses[grp]
            bead_pos[i] = (m[:, None] * positions[grp]).sum(axis=0) / m.sum()

        # CG distance matrix
        diff = bead_pos[:, None, :] - bead_pos[None, :, :]
        cg_dmat = np.sqrt((diff ** 2).sum(axis=-1))

        # Reference distance matrix (mean atom-atom distance per bead pair)
        ref_dmat = np.zeros((n_beads, n_beads))
        for i in range(n_beads):
            pi = positions[mapping[i]]
            for j in range(i + 1, n_beads):
                pj = positions[mapping[j]]
                d = np.sqrt(((pi[:, None, :] - pj[None, :, :]) ** 2).sum(axis=-1))
                ref_dmat[i, j] = ref_dmat[j, i] = d.mean()

        # RMSE on upper triangle
        triu = np.triu_indices(n_beads, k=1)
        diff_vals = cg_dmat[triu] - ref_dmat[triu]
        return float(np.sqrt(np.mean(diff_vals ** 2)))
