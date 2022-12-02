executable = RunScripts/i2i_pipeline.sh
arguments = "en-zh clipL transformer 2"
getenv = true
output = log/i2i_en-zh_clipL+transformer_seed2.out
error = log/i2i_en-zh_clipL+transformer_seed2.err
log = log/i2i_en-zh_clipL+transformer_seed2.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
