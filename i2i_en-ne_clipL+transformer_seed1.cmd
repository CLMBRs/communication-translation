executable = RunScripts/i2i_pipeline.sh
arguments = "en-ne clipL transformer 1"
getenv = true
output = log/i2i_en-ne_clipL+transformer_seed1.out
error = log/i2i_en-ne_clipL+transformer_seed1.err
log = log/i2i_en-ne_clipL+transformer_seed1.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue