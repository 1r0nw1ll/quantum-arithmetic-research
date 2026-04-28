Simplify `DatabaseOperations.execute_sql_flush()` by removing its redundant
`using` argument and inferring the database alias from `self.connection`.

Update the flush management command to call the simplified method signature.
