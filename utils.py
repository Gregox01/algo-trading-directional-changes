import pickle
from deap import base, creator, gp, tools

def save_checkpoint(data, filename):
    checkpoint = {
        'data': data,
    }

    # If data is a tuple containing a toolbox, we need to handle it specially
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], base.Toolbox):
        best_individual, toolbox = data
        toolbox_config = {
            k: getattr(v, '__name__', str(v)) for k, v in toolbox.__dict__.items()
        }
        checkpoint['data'] = (best_individual, toolbox_config)

    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(filename):
    try:
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)

        # Check if checkpoint is a tuple (old format) or a dict (new format)
        if isinstance(checkpoint, tuple):
            data = checkpoint
        elif isinstance(checkpoint, dict):
            data = checkpoint.get('data')
        else:
            raise ValueError("Unexpected checkpoint format")

        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
            best_individual, toolbox_config = data
            toolbox = base.Toolbox()
            for attr, value in toolbox_config.items():
                if value in dir(gp):
                    setattr(toolbox, attr, getattr(gp, value))
                elif value in dir(tools):
                    setattr(toolbox, attr, getattr(tools, value))
                else:
                    setattr(toolbox, attr, value)
            return best_individual, toolbox
        else:
            return data
    except (EOFError, pickle.UnpicklingError, ValueError) as e:
        print(f"Warning: Checkpoint file {filename} is empty, invalid, or in an unexpected format. Error: {e}")
        return None
