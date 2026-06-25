"""
Tests for the count-decoder negative-log-likelihood loss in
``terra.training.decode``.

Function under test
-------------------
``nb_nll(mu, theta, target, eps=1e-8)``
    Negative-binomial NLL, parameterized by mean ``mu`` and *inverse
    dispersion* ``theta``. Returns ``-log_prob.sum(dim=-1).mean()`` (sum
    over genes, mean over batch).

Oracle strategy
---------------
The references are INDEPENDENT of the implementation:

* We convert the ``(mu, theta)`` parameterization to
  ``torch.distributions.NegativeBinomial(total_count=theta,
  probs=mu/(mu+theta))``. That distribution has mean
  ``total_count * probs / (1 - probs) = mu`` and dispersion ``theta``,
  matching the convention used by the code under test.
* We cross-check against a second, fully hand-written gamma-function NB
  log-pmf. We never call the function under test to build its own oracle.
"""

import unittest

import torch

from terra.training.decode import nb_nll


def _torch_nb_log_prob(target, mu, theta):
    """Independent NB log-pmf via torch.distributions.

    Maps the (mu, theta) parameterization onto NegativeBinomial's
    (total_count, probs) parameterization.
    """
    probs = mu / (mu + theta)
    dist = torch.distributions.NegativeBinomial(total_count=theta, probs=probs)
    return dist.log_prob(target)


def _manual_nb_log_prob(x, mu, theta):
    """From-scratch NB log-pmf using the gamma-function closed form.

    P(X=x) = Gamma(x+theta)/(Gamma(theta)*x!) *
             (theta/(theta+mu))**theta * (mu/(theta+mu))**x
    Implemented in log space, written from the textbook PMF (not copied
    from the implementation under test).
    """
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )


class TestNBNLL(unittest.TestCase):
    def test_matches_torch_distribution_reference(self):
        # Small fixed batch (2 cells x 3 genes), well away from boundaries.
        mu = torch.tensor([[1.0, 2.0, 5.0],
                           [3.0, 0.5, 4.0]], dtype=torch.float64)
        theta = torch.tensor([2.0, 1.5, 3.0], dtype=torch.float64)
        target = torch.tensor([[0.0, 2.0, 4.0],
                              [1.0, 0.0, 6.0]], dtype=torch.float64)

        # Independent oracle: sum over genes (theta broadcasts over batch),
        # then mean over the batch dimension, negated.
        theta_b = theta.view(1, -1).expand_as(mu)
        ref_log_prob = _torch_nb_log_prob(target, mu, theta_b)
        expected = -ref_log_prob.sum(dim=-1).mean()

        got = nb_nll(mu, theta, target)

        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)

    def test_matches_manual_gamma_reference(self):
        # Cross-check against a second, fully hand-written oracle.
        mu = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
        theta = torch.tensor([1.0, 2.0], dtype=torch.float64)
        target = torch.tensor([[3.0, 0.0]], dtype=torch.float64)

        theta_b = theta.view(1, -1).expand_as(mu)
        expected = -_manual_nb_log_prob(target, mu, theta_b).sum(dim=-1).mean()

        got = nb_nll(mu, theta, target)

        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)

    def test_scalar_finite_and_nonnegative(self):
        mu = torch.tensor([[1.0, 2.0, 5.0]], dtype=torch.float64)
        theta = torch.tensor([2.0, 1.5, 3.0], dtype=torch.float64)
        target = torch.tensor([[0.0, 2.0, 4.0]], dtype=torch.float64)

        loss = nb_nll(mu, theta, target)

        self.assertEqual(loss.shape, torch.Size([]))  # scalar (mean reduction)
        self.assertTrue(torch.isfinite(loss))
        # NLL is a sum of per-gene -log(pmf); each term is >= 0.
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_gradients_flow_to_parameters(self):
        mu = torch.tensor([[1.0, 2.0, 5.0]],
                          dtype=torch.float64, requires_grad=True)
        theta = torch.tensor([2.0, 1.5, 3.0],
                             dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0.0, 2.0, 4.0]], dtype=torch.float64)

        loss = nb_nll(mu, theta, target)
        loss.backward()

        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(theta.grad)
        self.assertTrue(torch.isfinite(mu.grad).all())
        self.assertTrue(torch.isfinite(theta.grad).all())
        # The loss genuinely depends on mu, so gradients must be non-trivial.
        self.assertGreater(mu.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
