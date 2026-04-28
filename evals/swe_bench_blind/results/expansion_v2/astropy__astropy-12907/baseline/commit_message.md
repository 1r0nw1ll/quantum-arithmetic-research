Fix nested separability matrices for `&` compound models

Preserve the nested right-hand separability matrix when stacking `CompoundModel`
branches with `&`, and add a regression test for `Pix2Sky_TAN() &
(Linear1D() & Linear1D())`.
