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

    def test_revision_sources_exist(self):
        root = Path(__file__).resolve().parents[1]
        for relative in [
            "manuscript/main_revised_highlighted.tex",
            "manuscript/main_revised_clean.tex",
            "manuscript/response_to_reviewers.tex",
            "ARTICLE_FIGURE_MAP.md",
        ]:
            self.assertTrue((root / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()

