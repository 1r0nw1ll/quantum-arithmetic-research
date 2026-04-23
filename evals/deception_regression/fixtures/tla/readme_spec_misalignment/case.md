# Review Case: README claims more than the spec implements

Review this bundle. The README describes a two-phase commit protocol with a
coordinator and resource managers; the `.tla` has a single counter `round` and
nothing else. Does the harness catch the misalignment between what is claimed
and what is actually modeled?
