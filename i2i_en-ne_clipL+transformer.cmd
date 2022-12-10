seed_offset=0
seed=$(Process)
executable = RunScripts/i2i_pipeline.sh
arguments = "en-ne clipL transformer $(seed) $(seed_offset)"
getenv = true
output = log/i2i_en-ne_clipL+transformer_seed$(seed)+$(seed_offset).out
error = log/i2i_en-ne_clipL+transformer_seed$(seed)+$(seed_offset).err
log = log/i2i_en-ne_clipL+transformer_seed$(seed)+$(seed_offset).log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue 3
