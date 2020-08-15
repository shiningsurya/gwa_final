
# Special Topics in Physics: Gravitational Wave Data analysis

## Final project code


### Creating Datasets

There are three classes in the classification problem dealt here.
- Mixed sine 
- Chirp
- Core Collapse SuperNovae (CCSN)

`create_ms.py` creates mixed sine signals

`create_chirp.py` creates chirp signals

`create_ccsn.py` creates CCSN signals 

Each of the above scripts writes an `npy` file containing the Time-Frequency images of `118x90` shape.

### Training

`runner.py` script defines dataloader class, CNN model and trains.
