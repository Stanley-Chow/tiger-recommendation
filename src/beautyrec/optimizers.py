"""A readable, non-adaptive Sharpness-Aware Minimization (SAM) wrapper.

This is a PyTorch adaptation of the two-gradient algorithm in Foret et al.
(https://arxiv.org/abs/2010.01412), not the authors' JAX implementation.
The caller evaluates the same minibatch twice and clips only the second gradient.
"""

import math
from collections.abc import Iterator

import torch
from torch import nn


class SAM:
    """Use a base optimizer's exact parameters with an L2 gradient perturbation.

    ``first_step`` moves weights by ``rho * gradient / ||gradient||_2``. Compute
    a fresh gradient there, then call ``second_step`` to restore the original
    weights and apply the base optimizer to that second gradient. The wrapper
    deliberately stores copies of the original weights: subtracting perturbations
    would not restore their exact floating-point values.

    An exception during the caller's second forward/backward must call
    ``restore()`` in its exception handler. Both optimizer methods clean up their
    own failures. AdamW can be used as the base optimizer; AdamW's decoupled
    weight decay belongs to its final update, not to the perturbation gradient.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05) -> None:
        if not math.isfinite(rho) or rho < 0:
            raise ValueError("SAM rho must be finite and nonnegative.")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self._originals: dict[nn.Parameter, torch.Tensor] = {}
        self._pending = False
        parameters = list(self._parameters())
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("SAM requires each parameter to occur once in the base optimizer.")

    @property
    def param_groups(self) -> list[dict]:
        """Expose the live base groups, including their learning rates."""
        return self.base_optimizer.param_groups

    @property
    def pending(self) -> bool:
        """Whether a first pass is waiting for a second pass or restoration."""
        return self._pending

    def _parameters(self) -> Iterator[nn.Parameter]:
        for group in self.param_groups:
            yield from group["params"]

    def _finite_gradients(self) -> list[tuple[nn.Parameter, torch.Tensor]]:
        gradients = []
        for parameter in self._parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if gradient.is_sparse:
                raise ValueError("This SAM implementation requires dense gradients.")
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("SAM received a non-finite gradient.")
            gradients.append((parameter, gradient))
        return gradients

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> float:
        """Perturb weights; return the first gradient's global L2 norm.

        Missing or all-zero gradients and rho=0 make no parameter changes, but
        still open the two-pass operation. Non-finite gradients fail before any
        mutation. The norm uses float64 accumulation to avoid overflow of a norm
        made from otherwise finite float32 gradients.
        """
        if self._pending:
            raise RuntimeError("SAM first_step is already pending; second_step or restore it.")
        try:
            gradients = self._finite_gradients()
            norm = math.hypot(
                *(
                    float(torch.linalg.vector_norm(gradient, dtype=torch.float64))
                    for _, gradient in gradients
                )
            )
            if not math.isfinite(norm):
                raise FloatingPointError("SAM gradient norm is non-finite.")
            self._pending = True
            if self.rho > 0 and norm > 0:
                for parameter, gradient in gradients:
                    self._originals[parameter] = parameter.detach().clone()
                    # Normalize before scaling: rho / a tiny norm could overflow.
                    perturbation = (gradient.to(torch.float64) / norm * self.rho).to(parameter)
                    parameter.add_(perturbation)
                    if not bool(torch.isfinite(parameter).all()):
                        raise FloatingPointError("SAM perturbation made a parameter non-finite.")
            if zero_grad:
                self.zero_grad()
            return norm
        except BaseException:
            self.restore()
            raise

    @torch.no_grad()
    def restore(self, zero_grad: bool = True) -> None:
        """Cancel a pending perturbation, including parameters with no second gradient.

        Restoration is idempotent and does not update the base optimizer state.
        Pass zero_grad=False when preserving the second gradient for a base step.
        """
        for parameter, original in self._originals.items():
            parameter.copy_(original)
        self._originals.clear()
        self._pending = False
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        """Restore original weights, validate the second gradient, then update."""
        if not self._pending:
            raise RuntimeError("SAM second_step requires a preceding first_step.")
        try:
            # Restore first so even validation failures leave clean weights.
            self.restore(zero_grad=False)
            self._finite_gradients()
            self.base_optimizer.step()
        except BaseException:
            self.restore()
            raise
        if zero_grad:
            self.zero_grad()

    def state_dict(self) -> dict:
        """Checkpoint only complete updates; transient perturbations are never saved."""
        if self._pending:
            raise RuntimeError("Restore SAM parameters before saving optimizer state.")
        return {"rho": self.rho, "base_optimizer": self.base_optimizer.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        if self._pending:
            raise RuntimeError("Restore SAM parameters before loading optimizer state.")
        rho = float(state_dict["rho"])
        if not math.isfinite(rho) or rho < 0:
            raise ValueError("Saved SAM rho must be finite and nonnegative.")
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.rho = rho
