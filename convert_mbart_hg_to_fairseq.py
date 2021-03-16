import argparse
from EC_finetune.agents import CommunicationAgent



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ref Game Engine')
    parser.add_argument('--config', type=str)
