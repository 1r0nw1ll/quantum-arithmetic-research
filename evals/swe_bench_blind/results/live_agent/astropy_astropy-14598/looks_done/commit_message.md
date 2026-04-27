io.fits: preserve escaped quotes across CONTINUE parsing

Delay quote unescaping until long string fragments have been reassembled in
`Card._split()`, and add a regression test for doubled single-quotes near a
`CONTINUE` boundary.
