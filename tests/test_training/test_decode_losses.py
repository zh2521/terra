"""
Tests for the count-decoder negative-log-likelihood losses in
``terra.training.decode``.

Functions under test
---------------------
``nb_nll(mu, theta, target, eps=1e-8)``
    Negative-binomial NLL, parameterized by mean ``mu`` and *inverse
    dispersion* ``theta``. Returns ``-log_prob.sum(dim=-1).mean()`` (sum
    over genes, mean over batch).

``zinb_nll(mu, theta, zi_logits, target)``
    Zero-inflated NB NLL with the same ``(mu, theta)`` NB component plus a
    zero-inflation logit ``zi_logits`` (dropout logit). Returns
    ``-log_prob.sum(dim=-1).mean()``.

Oracle strategy
---------------
The references below are INDEPENDENT of the implementation:

* For NB we convert the ``(mu, theta)`` parameterization to
  ``torch.distributions.NegativeBinomial(total_count=theta,
  probs=mu/(mu+theta))``. That distribution has mean
  ``total_count * probs / (1 - probs) = mu`` and dispersion ``theta``,
  matching the scvi-tools convention used by the code under test.

* For ZINB we assemble the closed-form mixture log-pmf by hand from a
  from-scratch NB log-pmf (built with ``torch.lgamma``) and
  ``pi = sigmoid(zi_logits)``:
      x == 0 : log(pi + (1 - pi) * NB(0))
      x  > 0 : log(1 - pi) + log NB(x)
  We never call the functions under test to build their own oracle.
"""

import unittest

import torch


# Import from the source-mirroring module path.
from terra.training.decode import nb_nll, zinb_nll

# zinb_nll lazily uses scvi-tools (an optional, training-only dependency).
# Skip the ZINB tests when scvi is not installed (e.g. the default CI env);
# the pure-torch nb_nll tests always run.
try:
    import scvi  # noqa: F401
    _HAS_SCVI = True
except Exception:
    _HAS_SCVI = False


def _torch_nb_log_prob(target, mu, theta):
    """Independent NB log-pmf via torch.distributions.

    Maps the scvi (mu, theta) parameterization onto NegativeBinomial's
    (total_count, probs) parameterization.
    """
    probs = mu / (mu + theta)
    dist = torch.distributions.NegativeBinomial(total_count=theta, probs=probs)
    return dist.log_prob(target)


def _manual_nb_log_prob(x, mu, theta):
    """From-scratch NB log-pmf using the gamma-function closed form.

    P(X=x) = Gamma(x+theta)/(Gamma(theta)*x!) *
             (theta/(theta+mu))**theta * (mu/(theta+mu))**x
    Implemented in log space. This is written from the textbook PMF and is
    intentionally not copied from the implementation under test.
    """
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )


def _manual_zinb_log_prob(x, mu, theta, zi_logits):
    """From-scratch ZINB log-pmf (closed-form mixture)."""
    pi = torch.sigmoid(zi_logits)
    nb_lp = _manual_nb_log_prob(x, mu, theta)
    nb_pmf = torch.exp(nb_lp)
    zero_case = torch.log(pi + (1.0 - pi) * nb_pmf)
    nonzero_case = torch.log(1.0 - pi) + nb_lp
    return torch.where(x < 0.5, zero_case, nonzero_case)


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


@unittest.skipUnless(_HAS_SCVI, "scvi-tools not installed")
class TestZINBNLL(unittest.TestCase):
    def test_reduces_to_nb_when_zero_inflation_off(self):
        # zi_logits -> very negative => pi = sigmoid(zi_logits) ~ 0, so ZINB
        # collapses onto the plain NB.
        mu = torch.tensor([[1.0, 2.0, 5.0],
                           [3.0, 0.5, 4.0]], dtype=torch.float64)
        theta = torch.tensor([[2.0, 1.5, 3.0],
                              [2.0, 1.5, 3.0]], dtype=torch.float64)
        target = torch.tensor([[0.0, 2.0, 4.0],
                              [1.0, 0.0, 6.0]], dtype=torch.float64)
        zi_logits = torch.full_like(mu, -30.0)  # pi ~ 9e-14

        zinb_loss = zinb_nll(mu, theta, zi_logits, target)
        nb_loss = nb_nll(mu, theta[0], target)

        torch.testing.assert_close(zinb_loss, nb_loss, atol=1e-5, rtol=1e-5)

    def test_matches_closed_form_mixture_reference(self):
        # Independent closed-form ZINB oracle for a tiny input.
        mu = torch.tensor([[2.0, 4.0, 1.0]], dtype=torch.float64)
        theta = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        # Mixed zero / non-zero targets exercises both branches of the mixture.
        target = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64)
        zi_logits = torch.tensor([[0.5, -1.0, 1.0]], dtype=torch.float64)

        expected = -_manual_zinb_log_prob(
            target, mu, theta, zi_logits
        ).sum(dim=-1).mean()

        got = zinb_nll(mu, theta, zi_logits, target)

        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)

    def test_higher_zero_inflation_lowers_loss_on_observed_zeros(self):
        # All targets are zero. Increasing the zero-inflation probability
        # should make the model assign more mass to zeros, lowering the NLL.
        mu = torch.tensor([[2.0, 4.0, 1.0]], dtype=torch.float64)
        theta = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        target = torch.zeros_like(mu)

        low_zi = torch.full_like(mu, -2.0)   # pi ~ 0.119
        high_zi = torch.full_like(mu, 2.0)   # pi ~ 0.881

        loss_low = zinb_nll(mu, theta, low_zi, target)
        loss_high = zinb_nll(mu, theta, high_zi, target)

        self.assertLess(loss_high.item(), loss_low.item())

    def test_scalar_finite(self):
        mu = torch.tensor([[2.0, 4.0, 1.0]], dtype=torch.float64)
        theta = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        target = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64)
        zi_logits = torch.tensor([[0.5, -1.0, 1.0]], dtype=torch.float64)

        loss = zinb_nll(mu, theta, zi_logits, target)

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_gradients_flow_to_parameters(self):
        mu = torch.tensor([[2.0, 4.0, 1.0]],
                          dtype=torch.float64, requires_grad=True)
        theta = torch.tensor([[1.0, 2.0, 3.0]],
                             dtype=torch.float64, requires_grad=True)
        zi_logits = torch.tensor([[0.5, -1.0, 1.0]],
                                 dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64)

        loss = zinb_nll(mu, theta, zi_logits, target)
        loss.backward()

        for name, param in (("mu", mu), ("theta", theta), ("zi_logits", zi_logits)):
            with self.subTest(param=name):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
        # zi_logits must receive a real gradient signal.
        self.assertGreater(zi_logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
