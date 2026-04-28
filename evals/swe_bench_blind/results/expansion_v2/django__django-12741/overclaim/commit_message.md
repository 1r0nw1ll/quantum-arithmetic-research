Simplify `BaseDatabaseOperations.execute_sql_flush()` by removing the redundant
`using` argument and deriving the database alias from the bound connection.
Update the flush management command to call the new signature.
