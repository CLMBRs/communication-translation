## Code Guide 
We use `.yml` file configure the hyper-parameters.
- To run the EC training process: `sh run_training_wconfig.sh`.
- `src/playground.py` is the file that sythesizes all the parts.
- `src/bart_models.py (models.py)` is the central part of our engine, where
the speaker/listener is initialized by the BART model (RNN model).
- `src/modeling_bart.py` is the file that organize the BART model.
- `src/forward.py` takes the output of the model and calculates the loss.
 
