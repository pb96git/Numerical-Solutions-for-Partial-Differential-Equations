import inspect  # For extracting source code of functions
import joblib  # Efficient hashing for arrays
import hashlib  # SHA1 hashing
import numpy as np
import os
from Storage import Storage


#----------------------------------------------------------------
#                           Functions
#----------------------------------------------------------------
import numpy as np

def save_arrays(filename, **kwargs):
    """
    Save multiple arrays to a .npz file.

    Parameters
    ----------
    filename : str
        The name of the .npz file to save the arrays in.
    **kwargs : dict
        Keyword arguments where the keys are array names, and the values are the arrays to save.
    """
    np.savez(filename, **kwargs)

def load_arrays(filename):
    """
    Load arrays from a .npz file.

    Parameters
    ----------
    filename : str
        The name of the .npz file to load arrays from.

    Returns
    -------
    dict
        A dictionary containing the arrays, with keys as their names.
    """
    with np.load(filename) as data:
        return {key: data[key] for key in data}


def generate_hash(func1, func2, array1, array2, obj1, obj2):
    """Generate a hash string based on input data."""
    # Convert inputs into a tuple of strings/hashable data
    data = (
        inspect.getsource(func1),  # Source code of func1
        inspect.getsource(func2),  # Source code of func2
        joblib.hash(array1),       # Hash of array1
        joblib.hash(array2),       # Hash of array2
        str(obj1),                 # String representation of obj1
        str(obj2)                  # String representation of obj2
    )
    # Generate an SHA1 hash from the combined data
    hash_input = hashlib.sha1(str(data).encode('utf-8')).hexdigest()
    return hash_input
#----------------------------------------------------------------
#                           Examples 
#----------------------------------------------------------------
def savez_usage_example():
    # Simulation parameters
    time_step = 100
    filename = f"simulation_results_t{time_step:04d}.npz"  # Example: 'simulation_results_t0042.npz'

    # Generate data
    array_u = np.linspace(0, 1, 1000000)  # Array u for time step 42
    array_v = np.random.rand(1000000)     # Array v for time step 42

    # Save the arrays
    print(f"Saving arrays to {filename}...")
    save_arrays(filename, u=array_u, v=array_v)

    # Load the arrays
    print(f"Loading arrays from {filename}...")
    arrays = load_arrays(filename)

    # Access and verify the data
    print("First 5 elements of array 'u':", arrays['u'][:5])
    print("First 5 elements of array 'v':", arrays['v'][:5])

def joblib_usage_example():
    # Initialize the Storage system with a directory for caching
    storage = Storage(location='cache_dir', verbose=1)

    # Generate large arrays
    array1 = np.linspace(0, 1, 1000000)  # 1 million points evenly spaced between 0 and 1
    array2 = np.random.rand(1000000)     # 1 million random numbers

    # Save the arrays
    print("Saving arrays...")
    storage.save('array1', array1)
    storage.save('array2', array2)

    # Retrieve the arrays
    print("Retrieving arrays...")
    retrieved_array1 = storage.retrieve('array1')
    retrieved_array2 = storage.retrieve('array2')

    # Verify the data
    print("First 5 elements of retrieved_array1:", retrieved_array1[:5])
    print("First 5 elements of retrieved_array2:", retrieved_array2[:5])

def hash_example():
    
    x0 = 0.5
    func1 = lambda x: x**2
    func2 = lambda x: 0 if x <= x0 else 1
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    obj1 = {'key': 'value'}
    obj2 = [7, 8, 9]

    hash_string = generate_hash(func1, func2, array1, array2, obj1, obj2)
    print("Generated Hash:", hash_string)



if __name__ == '__main__':
    
    savez_usage_example()
    # joblib_usage_example()
    # hash_example()
    
    
    
   