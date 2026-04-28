timeseries: clarify required-column validation errors

Improve `BaseTimeSeries` required-column error messages so multi-column
requirements report the full required and found column lists consistently,
including when a removal leaves only a subset of the required prefix. Add a
regression test for removing the last non-time required column and update the
shared timeseries message expectations in sampled and binned tests.
