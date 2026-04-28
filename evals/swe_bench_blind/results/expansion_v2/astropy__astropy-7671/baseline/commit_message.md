Restore `minversion` preprocessing for `LooseVersion`

Reintroduce the version regex trimming that strips `dev`/`rc` suffixes
before comparing with `distutils.version.LooseVersion`, and add a
regression test for a `0.12dev` minimum version check.
