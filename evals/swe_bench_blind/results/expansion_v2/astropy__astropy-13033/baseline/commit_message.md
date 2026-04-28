Fix TimeSeries required-column error message

Update `BaseTimeSeries._check_required_columns()` so validation errors report
the full required and found column lists when multiple required columns are in
play, and add a regression test for removing a required non-time column.
