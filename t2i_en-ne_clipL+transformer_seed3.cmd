executable = RunScripts/t2i_pipeline.sh
arguments = "en-ne clipL transformer 3"
getenv = true
output = log/t2i_en-ne_clipL+transformer_seed3.out
error = log/t2i_en-ne_clipL+transformer_seed3.err
log = log/t2i_en-ne_clipL+transformer_seed3.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
