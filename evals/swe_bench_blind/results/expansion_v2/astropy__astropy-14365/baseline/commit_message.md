Fix QDP parsing for lowercase commands

Make the QDP line classifier treat commands case-insensitively so files with
lowercase `read serr` / `read terr` directives parse correctly. Add a
regression test that reads a lowercase-command QDP file through
`Table.read(..., format="ascii.qdp")`.
