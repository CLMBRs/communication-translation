executable = RunScripts/en-ne_hydra_pipeline_i2i_repro.sh
arguments = ""
getenv = true
output = log/pipeline_en-ne_i2i.out
error = log/pipeline_en-ne_i2i.err
log = log/pipeline_en-ne_i2i.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
