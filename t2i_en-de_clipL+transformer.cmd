seed_offset=0
seed=$(Process)
executable = RunScripts/t2i_pipeline.sh
arguments = "en-de clipL transformer $(seed) $(seed_offset)"
getenv = true
output = log/t2i_en-de_clipL+transformer_seed$(seed)+$(seed_offset).out
error = log/t2i_en-de_clipL+transformer_seed$(seed)+$(seed_offset).err
log = log/t2i_en-de_clipL+transformer_seed$(seed)+$(seed_offset).log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue 3
