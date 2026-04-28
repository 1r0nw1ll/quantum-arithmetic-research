Fixed migration autodetector crashes for model and field renames.

Use the renamed model's new key when looking up the target model state
during field rename detection, and add a regression test covering a
single-step model rename plus field rename.
