# Repair: fix in the wrong file

Patch is a unified diff with real code, but it modifies a contenttypes
helper instead of the UUIDField primary key path the canonical fix
touches. Decide.
