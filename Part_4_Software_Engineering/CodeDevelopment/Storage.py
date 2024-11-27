import joblib
import inspect
import os

class Storage:
    """
    Store large data structures (e.g., numpy arrays) efficiently using joblib.
    """

    def __init__(self, location='tmp', verbose=1):
        """
        Initialize the Storage object.

        Parameters
        ----------
        location : str
            Directory where objects are stored on disk.
        verbose : int
            Controls verbosity for joblib and this class.
        """
        # Initialize joblib's memory caching system with 'location'
        self.memory = joblib.Memory(location=location, verbose=verbose)
        self.verbose = verbose
        # Cache the retrieve method using joblib
        self.retrieve = self.memory.cache(self._retrieve, ignore=['data'])
        self.save = self.retrieve  # Alias for retrieve method

    def _retrieve(self, name, data=None):
        """
        Retrieve or save data based on the name.

        Parameters
        ----------
        name : str
            Identifier for the data to be retrieved or saved.
        data : Any, optional
            The data to save. If None, the data is retrieved instead.

        Returns
        -------
        Any
            The retrieved or saved data.
        """
        if self.verbose > 0:
            print(f"joblib save of {name}")
        return data
