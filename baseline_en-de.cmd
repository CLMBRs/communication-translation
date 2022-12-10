seed_offset=0
seed=$(Process)
executable = RunScripts/baseline.sh
arguments = "en-de $(seed) $(seed_offset)"
getenv = true
output = log/baseline_en-de_seed$(seed)+$(seed_offset).out
error = log/baseline_en-de_seed$(seed)+$(seed_offset).err
log = log/baseline_en-de_seed$(seed)+$(seed_offset).log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue 3
