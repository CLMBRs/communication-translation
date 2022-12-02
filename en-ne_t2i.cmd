executable = RunScripts/en-ne_hydra_pipeline_repro.sh
arguments = ""
getenv = true
output = log/pipeline_en-ne_t2i.out
error = log/pipeline_en-ne_t2i.err
log = log/pipeline_en-ne_t2i.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
