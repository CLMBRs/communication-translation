executable = RunScripts/i2i_pipeline.sh
arguments = "en-de clipL transformer 3"
getenv = true
output = log/i2i_en-de_clipL+transformer_seed3.out
error = log/i2i_en-de_clipL+transformer_seed3.err
log = log/i2i_en-de_clipL+transformer_seed3.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
