executable = RunScripts/baseline.sh
arguments = "en-ne 2"
getenv = true
output = log/baseline_en-ne_seed2.out
error = log/baseline_en-ne_seed2.err
log = log/baseline_en-ne_seed2.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
