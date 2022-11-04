from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict
import yaml
import argparse
import os

def flatten_hydra_cfg(hydra_cfg: DictConfig, config_prefix="") -> Dict:
    # This function assumes the config name are unique
    ret = {}
    for k, v in hydra_cfg.items():
        if not isinstance(v, DictConfig):
            ret[k] = v
        else:
            prefix = f"{k}" if config_prefix == "" else f"{config_prefix}.{k}"
            sub_config = flatten_hydra_cfg(v, prefix)
            # Note: there might be some issue here if key name is not unique
            for sub_k, sub_v in sub_config.items():
                if sub_k in ret:
                    print(f"{prefix}.{sub_k} is a duplicate")
                # ret[sub_k] = sub_v
            ret.update(sub_config)
            
    return ret
            

if __name__ == "__main__":
    # Get path to two config files
    parser = argparse.ArgumentParser(description="Check Script")
    parser.add_argument('--hydra_config', type=str)
    parser.add_argument('--flat_config', type=str)
    args = parser.parse_args()
    
    # Read Flat config
    flat_config = yaml.load(open(args.flat_config, "r"))
    
    # Infer the pipeline section specified by this config
    section_name = None
    if "backtranslate" in args.hydra_config:
        section_name = "backtranslate"
    elif "ec" in args.hydra_config:
        section_name = "ec"
    assert section_name is not None
    
    # Read Hydra config
    config_dir = os.path.dirname(args.hydra_config)
    initialize(config_path="../Configs")
    cfg = compose(config_name=args.hydra_config,)
    cfg = cfg[section_name]
    # Flatten hydra config
    flattened_hydra_cfg = flatten_hydra_cfg(cfg)
    
    # diff two configs:
    print("Diff:")
    for k, v in flat_config.items():
        if k not in flattened_hydra_cfg:
            print(f"`{k}` is not in hydra config")
            print()
        elif v != flattened_hydra_cfg[k]:
            print(f"(flat) >>> {k}: {v}")
            print(f"(hydra)<<< {k}: {flattened_hydra_cfg[k]}")
            print()
        continue
    # print(OmegaConf.to_yaml(cfg))