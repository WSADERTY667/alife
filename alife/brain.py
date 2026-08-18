# Brain module - Spiking Neural Network for ALife simulation

import numpy as np
from alife import config
from alife.utils import clamp


class Brain:
    """
    Spiking neural network brain with reward-modulated STDP learning.
    """

    def __init__(self, genome, n_hidden, parent_weights=None):
        self.n_hidden = n_hidden
        self.n_in = config.INPUT_SIZE
        self.n_out = config.OUTPUT_SIZE
        self.n = self.n_in + self.n_hidden + self.n_out
        self.hidden_slice = slice(self.n_in, self.n)

        self.v = np.zeros(self.n, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=np.float32)
        self.out_rate = np.zeros(self.n_out, dtype=np.float32)

        rng = np.random.default_rng()

        # Sparse-ish random connectivity, but still dense matrix in MVP.
        self.mask = rng.random((self.n, self.n)) < genome["conn_prob"]
        np.fill_diagonal(self.mask, False)

        # Guarantee that inputs and outputs are not completely isolated.
        in_mask = rng.random((self.n_in, self.n)) < max(genome["conn_prob"], 0.12)
        self.mask[:self.n_in, :] |= in_mask

        out_mask = rng.random((self.n, self.n_out)) < max(genome["conn_prob"], 0.15)
        self.mask[:, -self.n_out:] |= out_mask

        np.fill_diagonal(self.mask, False)

        self.W = (
            rng.normal(0.0, genome["weight_scale"], (self.n, self.n)).astype(np.float32)
            * self.mask
        )

        # Lamarckian inheritance: partially copy learned parent weights.
        if parent_weights is not None and parent_weights.shape == self.W.shape:
            lam = clamp(genome["lamarckian_weight"], 0.0, 1.0)
            self.W = ((1.0 - lam) * self.W + lam * parent_weights).astype(np.float32)
            self.W *= self.mask

        self.decay_base = genome["membrane_decay"]
        self.threshold_base = genome["threshold"]
        self.stdp_rate = genome["stdp_rate"]
        self.max_w = genome["weight_max"]

        if config.LEARNING:
            self.E = np.zeros((self.n, self.n), dtype=np.float32)
        else:
            self.E = None

    def step(self, sensors, mod):
        sensors = np.asarray(sensors, dtype=np.float32)
        pre = self.spikes

        # Synaptic current from previous spikes.
        current = pre @ self.W

        arousal = mod.get("arousal", 0.0)
        decay = clamp(self.decay_base + arousal * 0.02, 0.50, 0.99)
        threshold = clamp(self.threshold_base - arousal * 0.05, 0.30, 2.0)

        # Membrane update.
        self.v = self.v * decay + current * config.SYNAPTIC_SCALE

        # High stress adds neural noise.
        if arousal > 0.8:
            noise_size = self.n - config.INPUT_SIZE
            noise = np.random.normal(
                0.0,
                (arousal - 0.8) * 0.02,
                noise_size,
            ).astype(np.float32)
            self.v[config.INPUT_SIZE:] += noise

        new_spikes = np.zeros(self.n, dtype=np.float32)

        hidden_v = self.v[self.hidden_slice]
        fired = hidden_v >= threshold
        new_spikes[self.hidden_slice] = fired.astype(np.float32)
        hidden_v[fired] = 0.0

        # Input layer is rate-coded from sensors.
        new_spikes[:config.INPUT_SIZE] = np.clip(sensors, 0.0, 1.0)

        # Output motor rates.
        self.out_rate = 0.75 * self.out_rate + 0.25 * new_spikes[-self.n_out:]

        # Reward-modulated STDP.
        if config.LEARNING and self.E is not None:
            learn_rate = clamp(
                mod.get("plasticity", 0.0) * mod.get("dopamine", 0.0),
                -2.0,
                2.0,
            )

            if abs(learn_rate) > 1e-6:
                post = new_spikes
                delta = (
                    np.outer(pre, post) - np.outer(post, pre)
                ).astype(np.float32)

                self.E = self.E * 0.95 + delta * self.stdp_rate
                self.W += learn_rate * self.E * self.mask
                self.W = np.clip(self.W, -self.max_w, self.max_w)
                self.W *= self.mask

        self.spikes = new_spikes
        return self.out_rate
