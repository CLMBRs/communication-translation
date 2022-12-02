executable = RunScripts/en-ne_hydra_pipeline_repro_i2i_sameT2IHyper_all15.sh
arguments = "clipL transformer"
getenv = true
output = log/pipeline_en-ne_i2i_sameT2IHyper_clipL+transformer_all15.out
error = log/pipeline_en-ne_i2i_sameT2IHyper_clipL+transformer_all15.err
log = log/pipeline_en-ne_i2i_sameT2IHyper_clipL+transformer_all15.log
Requirements = (( machine == "patas-gn3.ling.washington.edu"  ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
