from pathlib import Path
import unittest

import numpy as np

from gasdynbench.modeling import bounded_from_logit


class RepositoryContractTests(unittest.TestCase):
    def test_bounded_transform(self):
        latent = np.array([-1e6, -2.0, 0.0, 2.0, 1e6])
        values = bounded_from_logit(latent, 1.0, 5.0)
        self.assertTrue(np.all(values >= 1.0))
        self.assertTrue(np.all(values <= 5.0))

    def test_scaled_mlp_analytical_jacobian(self):
        from gasdynbench.modeling import ScaledMLP

        rng = np.random.default_rng(7)
        x = rng.uniform(-1.0, 1.0, (80, 2))
        y = np.sin(x[:, 0]) + 0.3 * x[:, 1] ** 2
        model = ScaledMLP(hidden=(8, 8), seed=7, max_iter=300).fit(x, y)
        points = np.array([[-0.4, 0.2], [0.3, -0.6]])
        analytic = model.jacobian(points)[:, 0, :]
        step = 1.0e-6
        finite = np.empty_like(analytic)
        for j in range(points.shape[1]):
            offset = np.zeros_like(points)
            offset[:, j] = step
            finite[:, j] = (model.predict(points + offset) - model.predict(points - offset)) / (2.0 * step)
        np.testing.assert_allclose(analytic, finite, rtol=2.0e-5, atol=2.0e-6)

    def test_revision_sources_exist(self):
        root = Path(__file__).resolve().parents[1]
        for relative in [
            "manuscript/main.tex",
            "manuscript/main_clean.tex",
            "manuscript/ref.bib",
            "manuscript/response_to_reviewers.tex",
            "ARTICLE_FIGURE_MAP.md",
            "scripts/export_original_article_figures.py",
            "scripts/run_application_audits.py",
            "scripts/run_high_dimensional_scaling.py",
            "src/gasdynbench/application_audits.py",
            "src/gasdynbench/high_dimensional.py",
            "docs/workflows/problem_workflows.tex",
            "docs/workflows/problem_workflows.pdf",
            "results/revision/nozzle_gradient_audit.csv",
            "results/revision/shock_tube_many_query.csv",
        ]:
            self.assertTrue((root / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
