from scipy.stats import linregress
import numpy as np

def fit_in_best_fit_interval(f: np.array[float], times: np.array[float], start: int = None, end: int = None):
    """
    Finds the best fit interval for a given "MSD" data set. 
    The best fit interval is the interval that minimizes the mean absolute error of the slope of the linear regression of the log-log plot of the "MSD" data.
    :param f: array[float], "MSD" data
    :param start: int, time index at which to start fitting
    :param end: int, time index at which to end fitting
    :param times: array[float], times at which position data was collected in the simulation
    :return best_fit_interval: int, time index at which to end fitting
    """
    best_fit_interval = 0
    min_mae = 100000
    int_len = int((end - start)/100)
    
    scale = 1
    slope_tol = 0.5 # Tolerance for the slope of the linear regression, i.e., the slope of the best fit line must be within 0.5 of 1.
    
    while scale < 10: # Downscale the interval length by powers of 2 to find the best fit interval. Anything more than 10 (1/10000th trajectory length) will be likely to be too small an interval to be useful.
        intervals = np.linspace(start, end, int_len, dtype=int)
        for i in range(len(intervals)-1):
            slope, intercept, r_value, p_value, std_err = linregress(np.log(times[intervals[i]: intervals[i+1]]), np.log(f[intervals[i]:intervals[i+1]]))
            mae = np.abs(slope - 1)
            if mae < min_mae and np.abs(slope - 1) < slope_tol:
                best_fit_interval = i
                min_mae = mae
                break
        scale *= 2

    if min_mae == 100000:  # If no good fit is found, return the entire interval as the best fit, since this is the best we can do.
        return start, end, min_mae
    
    slope, intercept, r_value, p_value, std_err = linregress(times[start:end], f[start:end])
    
    fit_dict = {'fit_err': std_err, 'slope_err': min_mae, 'interval': [intervals[best_fit_interval], intervals[best_fit_interval+1]]}
    
    return slope, fit_dict

def fit_with_blockavg(f: np.array[float], times: np.array[float] = None, start: int = None, end: int = None):
    """
    Compute block average of MSD data, excluding initial steps. Theoretically should lead to lower uncertainity in the fit.

    Parameters:
        msd: array-like, Mean Squared Displacement values.
        block_length: int, length of each block.
        initial_steps: int, number of initial steps to exclude.
    
    Returns:
        block_averages: array-like, block-averaged MSD values.
    """
    
    f = f[start:end]
    scale = 20
    block_lengths = np.arange(1, len(f)//2, len(f)//scale)
    min_mae = 100000

    for i, block_length in enumerate(block_lengths):
        num_blocks = len(f) // block_length
        block_averages = np.zeros(num_blocks)
        time_averages = np.zeros(num_blocks)
        
        for i in range(num_blocks):
            block_averages.append(np.mean(f[i * block_length : (i + 1) * block_length]))
            time_averages.append(np.mean(times[i * block_length : (i + 1) * block_length]))
        
        slope, intercept, r_value, p_value, std_err = linregress(np.log(time_averages), np.log(block_averages))
        if np.abs(slope - 1) < min_mae:
            slope, intercept, r_value, p_value, std_err = linregress(time_averages, block_averages)
            min_mae = np.abs(slope - 1)
            break
    
    if min_mae == 100000:
        slope, intercept, r_value, p_value, std_err = linregress(times[start:end], f[start:end])
    
    fit_dict = {'fit_err': std_err, 'slope_err': min_mae, 'block_length': block_length}
    return slope, fit_dict

def fit_data(f: np.array[float], times: np.array[float], start: int = None, end: int = None, smoothing: str = 'best_fit'):
    """
    Perform a linear regression.
    :param f: array[float], "MSD" or time correlation data
    :param start: int, time index at which to start fitting
    :param end: int, time index at which to end fitting
    :param times: array[float], times at which position data was collected in the simulation
    :return lij: float, transport coefficient, i.e., slope of "MSD" in fitting region
    """
    
    if smoothing == 'best_fit':
        lij, fit_dict = fit_in_best_fit_interval(f, times, start, end)
    
    elif smoothing == 'blockavg':
        lij, fit_dict = fit_with_blockavg(f, times, start, end)
    
    else:
        slope, intercept, r_value, p_value, std_err = linregress(np.log(times[start:end]), np.log(f[start:end]))
        slope_err = np.abs(slope - 1)
        lij, intercept, r_value, p_value, std_err = linregress(times[start:end], f[start:end])
        fit_dict = {'fit_err': std_err, 'slope_err': slope_err}
    
    return lij, fit_dict