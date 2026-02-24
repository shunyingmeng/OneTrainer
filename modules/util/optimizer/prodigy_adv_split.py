import math
import warnings
from statistics import harmonic_mean

import torch
import torch.distributed as dist

from adv_optm import Prodigy_adv


class ProdigyAdvSplitGroups(Prodigy_adv):
    """
    Prodigy ADV with split_groups support.

    When split_groups=True, each parameter group maintains its own adaptive
    learning rate multiplier (d value), allowing UNet, text encoders, and
    other components to independently discover their optimal learning rates.

    When split_groups_mean=True, each group still tracks independent Prodigy
    statistics, but parameter updates use the harmonic mean of all group d
    values. This provides a middle ground between fully shared and fully
    independent learning rates.
    """

    def __init__(
        self,
        *args,
        split_groups: bool = True,
        split_groups_mean: bool = False,
        **kwargs,
    ):
        # Force disable compiled_optimizer when split_groups is active.
        # torch.compile captures tensor references at trace time, breaking
        # the accumulator swapping mechanism.
        if split_groups and kwargs.get('compiled_optimizer', False):
            warnings.warn(
                "split_groups is incompatible with compiled_optimizer. "
                "Disabling compiled_optimizer.",
                stacklevel=2,
            )
            kwargs['compiled_optimizer'] = False

        self._split_groups = split_groups
        self._split_groups_mean = split_groups_mean

        # Parent __init__ calls init_step(), which we override.
        # Set flag so our init_step knows we're still in __init__.
        self._initialized = False

        super().__init__(*args, **kwargs)

        # Auto-disable split_groups for single param_group
        if self._split_groups and len(self.param_groups) == 1:
            print("Optimizer contains a single param_group -- 'split_groups' has been disabled.")
            self._split_groups = False

        if self._split_groups:
            # Create per-group accumulators
            for group in self.param_groups:
                p = group['params'][0]
                group['running_d_numerator'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                group['running_d_denom'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)

            # Initialize the class-level accumulators to prevent parent's lazy init
            # (step_parameter checks `hasattr(self, 'd_denom')`)
            device = self.param_groups[0]['params'][0].device
            self.d_denom = torch.tensor(0.0, device=device)
            self.d_numerator = torch.tensor(0.0, device=device)

        self._initialized = True

    def init_step(self):
        """Reset per-group accumulators for the upcoming step."""
        if not getattr(self, '_initialized', False) or not self._split_groups:
            # During __init__ or when split_groups is off, use parent behavior
            super().init_step()
            return

        # Set betas that parent code needs (mirrors parent's init_step)
        g_group = self.param_groups[0]
        self.beta1, self.beta2_default = g_group['betas']
        self.beta3 = g_group['beta3']
        if self.beta3 is None:
            self.beta3 = math.sqrt(self.beta2_default)

        # Reset per-group accumulators
        for group in self.param_groups:
            running_num = group['running_d_numerator']
            running_den = group['running_d_denom']
            running_den.zero_()
            running_num.fill_(group.get('d_numerator', 0.0) * self.beta3)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if not self._split_groups:
            super().step_parameter(p, group, i)
            return

        if p.grad is None:
            return

        # Ensure per-group accumulators are on the right device
        running_num = group['running_d_numerator']
        running_den = group['running_d_denom']
        if running_num.device != p.device:
            group['running_d_numerator'] = running_num = running_num.to(p.device)
        if running_den.device != p.device:
            group['running_d_denom'] = running_den = running_den.to(p.device)

        # Swap class-level accumulators to this group's tensors.
        # _step_parameter() accumulates via .add_() on self.d_numerator/d_denom,
        # so in-place ops will go to the per-group tensors.
        self.d_numerator = running_num
        self.d_denom = running_den

        if self._split_groups_mean:
            # In mean mode, parameter updates use shared_d (harmonic mean)
            # while Prodigy stats still accumulate per-group.
            shared_d = getattr(self, '_shared_d', None)
            if shared_d is not None:
                saved_d = group['d']
                group['d'] = shared_d
                super().step_parameter(p, group, i)
                group['d'] = saved_d
            else:
                # First step: shared_d not yet computed, use per-group d
                super().step_parameter(p, group, i)
        else:
            super().step_parameter(p, group, i)

    def calculate_d(self):
        """Calculate d independently for each parameter group."""
        if not self._split_groups:
            super().calculate_d()
            return

        for group in self.param_groups:
            prodigy_active = not (group.get('prodigy_steps', 0) > 0 and group['k'] >= group['prodigy_steps'])

            if prodigy_active:
                d_max = group['d_max']
                d_coef = group['d_coef']
                growth_rate = group['growth_rate']

                running_num = group['running_d_numerator']
                running_den = group['running_d_denom']

                if self.fsdp_in_use and dist.is_available() and dist.is_initialized():
                    dist_tensor = torch.stack([running_num, running_den])
                    dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                    global_d_numerator = dist_tensor[0].item()
                    global_d_denom = dist_tensor[1].item()
                else:
                    global_d_numerator = running_num.item()
                    global_d_denom = running_den.item()

                d_hat = group['d']
                if global_d_denom > 0:
                    d_hat = d_coef * global_d_numerator / global_d_denom
                    if group.get('d_limiter', False):
                        d_hat = min(group['d'] * (2 ** 0.25), d_hat)
                    if group['d'] == group['d0']:
                        group['d'] = max(group['d'], d_hat)
                    d_max = max(d_max, d_hat)
                    group['d'] = min(d_max, group['d'] * growth_rate)

                group['d_numerator'] = global_d_numerator
                group['d_max'] = d_max

            # Increment step counter
            group['k'] += 1

        # Compute harmonic mean if split_groups_mean is enabled
        if self._split_groups_mean:
            d_values = [group['d'] for group in self.param_groups if group['d'] > 0]
            if d_values:
                self._shared_d = harmonic_mean(d_values)
            else:
                self._shared_d = None
