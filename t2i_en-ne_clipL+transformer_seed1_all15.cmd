executable = RunScripts/t2i_pipeline_all15.sh
arguments = "en-ne clipL transformer 1"
getenv = true
output = log/t2i_en-ne_clipL+transformer_seed1_all15.out
error = log/t2i_en-ne_clipL+transformer_seed1_all15.err
log = log/t2i_en-ne_clipL+transformer_seed1_all15.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
