executable = RunScripts/baseline.sh
arguments = "en-si 3"
getenv = true
output = log/baseline_en-si_seed3.out
error = log/baseline_en-si_seed3.err
log = log/baseline_en-si_seed3.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue