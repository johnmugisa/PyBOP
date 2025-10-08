import logging
import numpy as np
# def extract_model_data(problem, parameter_values=None):

#     if parameter_values is None:
#         parameter_values = problem.x0  # Use initial values from the problem

#     # Extract time data from the dataset
#     time_data = problem.dataset.get("Time [s]")
#     if time_data is None:
#         raise KeyError("Time data 'Time [s]' not found in the dataset.")

#     # Evaluate the model with the given parameters
#     model_output = problem.evaluate(parameter_values)

#     # Debugging logs for model_output
#     logging.info(f"Type of model_output: {type(model_output)}")
#     if isinstance(model_output, dict):
#         logging.info(f"Keys in model_output: {list(model_output.keys())}")
#     elif isinstance(model_output, np.ndarray):
#         logging.info(f"Model output array shape: {model_output.shape}")
#     else:
#         logging.warning(f"Unexpected model_output type: {type(model_output)}")

#     # Ensure model_output is a dictionary
#     if not isinstance(model_output, (dict, list, np.ndarray)):
#         raise TypeError(f"model_output must be dict, list, or ndarray, got {type(model_output)}.")

#     # Extract target output, if available
#     target_output = problem.get_target() if hasattr(problem, 'get_target') else {}

#     # Debugging logs for target_output
#     logging.info(f"Type of target_output: {type(target_output)}")
#     if isinstance(target_output, dict):
#         logging.info(f"Keys in target_output: {list(target_output.keys())}")
#     elif isinstance(target_output, np.ndarray):
#         logging.info(f"Target output array shape: {target_output.shape}")
#     else:
#         logging.warning(f"Unexpected target_output type: {type(target_output)}")

#     # Initialize data dictionary with time data
#     data_dict = {'Time [s]': time_data}

#     # Validate that problem.signal is iterable
#     if not isinstance(problem.signal, (list, tuple)):
#         raise TypeError(f"Expected 'problem.signal' to be a list or tuple, got {type(problem.signal)}")

#     # Process signals
#     for signal in problem.signal:
#         logging.debug(f"Processing signal: {signal}")
#         # Handle model_output
#         model_signal_output = model_output.get(signal, []) if isinstance(model_output, dict) else []
#         if isinstance(model_signal_output, (int, float)):
#             model_signal_output = [model_signal_output]
#         elif isinstance(model_signal_output, np.ndarray):
#             model_signal_output = model_signal_output.tolist()
#         elif not isinstance(model_signal_output, list):
#             model_signal_output = []

#         # Handle target_output
#         target_signal_output = target_output.get(signal, []) if isinstance(target_output, dict) else []
#         if isinstance(target_signal_output, (int, float)):
#             target_signal_output = [target_signal_output]
#         elif isinstance(target_signal_output, np.ndarray):
#             target_signal_output = target_signal_output.tolist()
#         elif not isinstance(target_signal_output, list):
#             target_signal_output = []

#         # Store signal outputs in data_dict
#         data_dict[signal] = model_signal_output

#     return data_dict

def extract_model_data(problem, parameter_values=None):
    # Set default parameter values if not provided
    if parameter_values is None:
        parameter_values = problem.x0

    # Evaluate the model
    model_output = problem.evaluate(parameter_values)

    # Assume model_output is a dictionary with keys for each variable
    # Extract time data
    time_data = model_output.get('Time [s]')
    if time_data is None:
        time_data = problem.dataset.get('Time [s]')
        if time_data is None:
            raise KeyError("Time data 'Time [s]' not found.")

    # Extract desired signals
    signals = problem.signal if hasattr(problem, 'signal') else []
    data_dict = {'Time [s]': time_data}
    for signal in signals:
        data_dict[signal] = model_output.get(signal)

    return data_dict

# def get_model_data(problem, parameter_values=None):
#     """
#     Evaluate the model with given parameters and extract time and voltage data.
#     """
#     # Prepare inputs by combining default parameters with any provided parameter_values
#     inputs = problem.parameters.as_dict()
#     if parameter_values is not None:
#         inputs.update(parameter_values)

#     # Evaluate the model with the given inputs
#     model_output = problem.evaluate(inputs)

#     # Extract time data from the dataset
#     # Use problem._dataset instead of problem.dataset.data
#     time_data = problem._dataset.get('Time [s]')
#     if time_data is None:
#         raise KeyError("Time data 'Time [s]' not found in the dataset.")

#     # Attempt to extract 'Desired Voltage' from model_output
#     voltage_data = model_output.get('Desired Voltage', None)
#     if voltage_data is None:
#         # Fall back to 'Voltage [V]' if 'Desired Voltage' is not available
#         voltage_data = model_output.get('Voltage [V]', None)
#     if voltage_data is None:
#         raise KeyError("Voltage data not found in model output.")

#     data_dict = {
#         'Time [s]': time_data,
#         'Voltage [V]': voltage_data
#     }
#     return data_dict


##modified to process initial values before optimisation

def get_model_data(problem, parameter_values=None):
    """
    Evaluate the model with given parameters and extract time and voltage data.
    """
    # Prepare inputs by combining default parameters with any provided parameter_values
    inputs = problem.parameters.as_dict()
    if parameter_values is not None:
        inputs.update(parameter_values)

    # Evaluate the model with the given inputs
    try:
        model_output = problem.evaluate(inputs)
    except Exception as e:
        raise RuntimeError(f"Model evaluation failed: {e}")

    # Extract time data from the dataset
    time_data = problem._dataset.get('Time [s]')
    if time_data is None:
        raise KeyError("Time data 'Time [s]' not found in the dataset.")

    # Attempt to extract 'Desired Voltage' from model_output
    voltage_data = model_output.get('Desired Voltage', None)
    if voltage_data is None:
        # Fall back to 'Voltage [V]' if 'Desired Voltage' is not available
        voltage_data = model_output.get('Voltage [V]', None)
    if voltage_data is None:
        raise KeyError("Voltage data not found in model output.")

    return {
        'Time [s]': time_data,
        'Voltage [V]': voltage_data
    }