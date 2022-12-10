seed_offset=1
seed=$(Process)
executable = echo.sh
arguments = "en-de clipL transformer $(seed) $(seed_offset)"
getenv = true
output = log/dummy_for_playwithcode_$(seed)+$(seed_offset).out
error = log/dummy_for_playwithcode_$(seed)+$(seed_offset).err
log = log/dummy_for_playwithcode_$(seed)+$(seed_offset).log
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue 3
