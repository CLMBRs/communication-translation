model_dir=$1
host=xuhuiz@tir.lti.cs.cmu.edu:/home/xuhuiz/projects/xuhuiz/communication-translation/${model_dir}/
mkdir -p translation_output/${model_dir}/
scp -r ${host} translation_output/${model_dir}/../