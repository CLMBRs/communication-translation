executable = RunScripts/t2i_pipeline.sh
arguments = "en-de clipL transformer 2"
getenv = true
output = log/t2i_en-de_clipL+transformer_seed2.out
error = log/t2i_en-de_clipL+transformer_seed2.err
log = log/t2i_en-de_clipL+transformer_seed2.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
