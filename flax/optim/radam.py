# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flax.struct as struct

import jax.numpy as jnp
from jax import lax

import numpy as onp

from flax.optim.base import OptimizerDef


# Unfortunately, this means that we have to use numpy to store internal state, which makes it pretty slow.
@struct.dataclass
class _RAdamHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray


@struct.dataclass
class _RAdamParamState:
    exp_avg: onp.ndarray
    exp_avg_sq: onp.ndarray


class RAdam(OptimizerDef):
    """Adam optimizer."""

    def __init__(self,
                 learning_rate=None,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 weight_decay=0.0):
        """Constructor for the Adam optimizer.
        Args:
          learning_rate: the step size used to update the parameters.
          beta1: the coefficient used for the moving average of the
            gradient (default: 0.9).
          beta2: the coefficient used for the moving average of the
            gradient magnitude (default: 0.999).
          eps: the term added to the gradient magnitude estimate for
            numerical stability.
          weight_decay: AdamW style weight decay rate
            (relative to learning rate).
        """
        hyper_params = _RAdamHyperParams(learning_rate, beta1, beta2, eps,
                                         weight_decay)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _RAdamParamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        lr = hyper_params.learning_rate
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay
        t = step + 1.
        rho_inf = 2.0 / (1 - beta2) - 1

        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq
        beta2_t = beta2 ** 5
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2_t)
        rho_t = rho_inf - 2.0 * t * beta2_t / (1 - beta2_t)
        if rho_t <= 5:
            step_size = 1.0 / (1 - beta1 ** t)
        else:
            step_size = lax.sqrt((1 - beta2_t) * (rho_t - 4) / (rho_inf - 4) *
                                 (rho_t - 2) / rho_t * rho_inf / (rho_inf - 2)) / (1 - beta1 ** t)

        if rho_t <= 5:
            new_param = param - lr * step_size * grad_ema
            new_param -= lr * weight_decay * param
        else:
            denom = lax.sqrt(grad_sq_ema_corr) + hyper_params.eps
            new_param = param - lr * step_size * grad_ema / denom
            new_param -= lr * weight_decay * param
        new_state = _RAdamParamState(grad_ema, grad_sq_ema)
        return new_param, new_state
