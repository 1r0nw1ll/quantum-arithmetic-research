Fixed `call_command()` handling for required mutually exclusive groups.

Include kwargs-backed options from required mutually exclusive groups when
building the parser input, and add a regression test covering
`call_command(..., shop_id=1)`.
