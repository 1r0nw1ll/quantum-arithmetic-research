Honor required mutually exclusive kwargs in call_command

Ensure `call_command()` passes keyword-supplied options from required
mutually exclusive parser groups through `parse_args()` when simulating CLI
parsing. Add a regression test covering a command whose required group is
satisfied with `shop_id=...`.
