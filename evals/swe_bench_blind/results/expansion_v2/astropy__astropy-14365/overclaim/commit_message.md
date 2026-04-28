Fix QDP reader handling of lowercase commands

Parse QDP command lines case-insensitively so files using lowercase
`read serr`/`read terr` are accepted. Add a regression test covering a
lowercase `read serr` header with error columns.
