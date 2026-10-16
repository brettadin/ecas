import numpy as np

from petitRADTRANS.retrieval.utils import log_prior, uniform_prior, gaussian_prior, log_gaussian_prior, delta_prior


class Parameter:
    r"""Allow easy translation between the pyMultinest hypercube and the physical unit space.

    Each parameter includes a name, which can be used
    as a reference in the model function, a value, a flag of whether it's a free parameter,
    and if it's free, a function that translates the unit hypercube into physical space.
    The remainder of the arguments deal with the corner plots.

    Args:
        name : string
            The name of the parameter. Must match the name used in the model function.
        is_free_parameter : bool
            True if the parameter should be sampled in the retrieval
        value : float
            The value of the parameter. Set using set_param.
        transform_prior_cube_coordinate : method
            Transform the unit interval [0,1] to the physical space of the parameter.
        plot_in_corner : bool
            True if this parameter should be included in the output corner plot
        corner_ranges : Tuple(float,float)
            The axis range of the parameter in the corner plot
        corner_transform : method
            A function to scale or transform the value of the parameter for prettier plotting.
        corner_label : string
            The axis label for the parameter, defaults to name.
    """

    def __init__(self,
                 name,
                 is_free_parameter,
                 value=None,
                 transform_prior_cube_coordinate=None,
                 plot_in_corner=False,
                 corner_ranges=None,
                 corner_transform=None,
                 corner_label=None):

        self.name = name
        self.is_free_parameter = is_free_parameter
        self.value = value
        self.transform_prior_cube_coordinate = \
            transform_prior_cube_coordinate
        self.plot_in_corner = plot_in_corner
        self.corner_ranges = corner_ranges
        self.corner_transform = corner_transform
        self.corner_label = corner_label

    def get_param_uniform(self, cube):
        if self.is_free_parameter:
            return self.transform_prior_cube_coordinate(cube)

        raise ValueError(f"Error! Parameter '{self.name}' is not a free parameter")

    def get_flattened_value(self, value=None):
        if value is None:
            value = self.value

        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            _value = []

            keys = list(value.keys())
            values = list(value.values())

            for i, v in enumerate(values):
                v = self.get_flattened_value(value=v)
                k = self.get_flattened_value(value=[keys[i]])
                flattened_dict = np.concatenate((k, v))
                _value.append(flattened_dict.flatten())
        else:
            _value = value

        return np.array(_value).flatten()

    def set_param(self, value):
        if self.is_free_parameter:
            self.value = value
            return

        raise ValueError(f"Error! Parameter '{self.name}' is not a free parameter")


class RetrievalParameter:
    """Used to set up retrievals.

    Stores the prior function. Prior parameters depends on the type of prior. e.g., for uniform and log prior, these
    are the bounds of the prior. For gaussian priors and alike, these are the values of the mean and full width
    half maximum.

    Args:
        name:
            name of the parameter to retrieve, must match the corresponding model parameter of a SpectralModel
        prior_parameters:
            list of two values for the prior parameters, depends on the prior type
        prior_type:
            type of prior to use, the available types are stored into available_priors
        custom_prior:
            function with arguments (cube, *args), args being positional arguments in prior_parameters
    """
    __available_priors = [
        'log',
        'uniform',
        'gaussian',
        'log_gaussian',
        'delta',
        'custom'
    ]

    def __init__(self, name, prior_parameters, prior_type='uniform', custom_prior=None):
        # Check prior parameters validity
        if not hasattr(prior_parameters, '__iter__'):
            raise ValueError(
                f"'prior_parameters' must be an iterable of size 2, but is of type '{type(prior_parameters)}'"
            )
        elif np.size(prior_parameters) < 2:
            raise ValueError(
                f"'prior_parameters' must be of size 2, but is of size '{np.size(prior_parameters)}'"
            )
        elif prior_parameters[0] > prior_parameters[1] and (prior_type == 'log' or prior_type == 'uniform'):
            raise ValueError(
                f"lower prior boundaries ({prior_parameters[0]}) "
                f"must be lower than upper prior boundaries ({prior_parameters[1]})"
            )

        self.name = name
        self.prior_parameters = prior_parameters
        self.prior_type = prior_type

        # Set prior
        if self.prior_type == 'log':
            def prior(x):
                return log_prior(
                    cube=x,
                    lx1=self.prior_parameters[0],
                    lx2=self.prior_parameters[1]
                )
        elif self.prior_type == 'uniform':
            def prior(x):
                return uniform_prior(
                    cube=x,
                    x1=self.prior_parameters[0],
                    x2=self.prior_parameters[1]
                )
        elif self.prior_type == 'gaussian':
            def prior(x):
                return gaussian_prior(
                    cube=x,
                    mu=self.prior_parameters[0],
                    sigma=self.prior_parameters[1]
                )
        elif self.prior_type == 'log_gaussian':
            def prior(x):
                return log_gaussian_prior(
                    cube=x,
                    mu=self.prior_parameters[0],
                    sigma=self.prior_parameters[1]
                )
        elif self.prior_type == 'delta':
            def prior(x):
                return delta_prior(
                    cube=x,  # actually useless
                    x1=self.prior_parameters[0],
                    x2=self.prior_parameters[1]  # actually useless
                )
        elif self.prior_type == 'custom':
            def prior(x):
                return custom_prior(
                    cube=x,
                    *prior_parameters
                )
        else:
            raise ValueError(
                f"prior type '{prior_type}' not implemented "
                f"(available prior types: {'|'.join(RetrievalParameter.__available_priors)})"
            )

        self.prior_function = prior

    @classmethod
    def from_dict(cls, dictionary):
        """Convert a dictionary into a list of RetrievalParameter.
        The keys of the dictionary are the names of the RetrievalParameter. The values of the dictionary must be
        dictionaries with keys 'prior_parameters' and 'prior_type'.

        Args:
            dictionary: a dictionary

        Returns:
            A list of RetrievalParameter.
        """
        new_retrieval_parameters = []

        for key, parameters in dictionary.items():
            new_retrieval_parameters.append(
                cls(
                    name=key,
                    prior_parameters=parameters['prior_parameters'],
                    prior_type=parameters['prior_type']
                )
            )

        return new_retrieval_parameters

    def put_into_dict(self, dictionary=None):
        """Convert a RetrievalParameter into a dictionary.

        Args:
            dictionary: a dictionary; if None, a new dictionary is created

        Returns:
            A dictionary.
        """
        if dictionary is None:
            dictionary = {}

        dictionary[self.name] = {
            'prior_boundaries': self.prior_parameters,
            'prior_type': self.prior_type
        }

        return dictionary
