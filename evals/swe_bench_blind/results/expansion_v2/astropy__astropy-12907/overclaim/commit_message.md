Fix nested compound separability handling in `&` compositions

Preserve the right-hand separability matrix when stacking nested compound models so existing separable structure is not collapsed into a fully coupled block. Add a regression test covering `Pix2Sky_TAN() & (Linear1D() & Linear1D())`.
