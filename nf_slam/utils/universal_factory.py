import dataclasses
import inspect
from inspect import signature

import jax.numpy as jnp
import numpy as np


class UniversalFactory:
    def __init__(self, classes):
        self._classes = {x.__name__: x for x in classes}

    def make(self, class_name, parameters, **kwargs):
        result = self._make_impl(class_name, parameters, **kwargs)
        return result

    def _make_impl(self, function, parameters, **kwargs):
        if type(parameters) != dict:
            return parameters
        if function is None:
            return parameters
        if function is int:
            return int(parameters)
        if function is float:
            return float(parameters)
        if function is str:
            return str(parameters)
        if function == np.array or function == np.ndarray:
            return np.array(parameters)
        if function == jnp.array or function == jnp.ndarray:
            return jnp.array(parameters)
        return self._make_from_function(function, parameters, **kwargs)

    def _make_from_function(self, function, parameters, **kwargs):
        function_arguments = signature(function).parameters
        function_parameters = {}
        for key, value in function_arguments.items():
            # noinspection PyProtectedMember
            if key in parameters.keys():
                child_parameters = parameters[key]
                annotation = value.annotation
                # noinspection PyProtectedMember
                if annotation is inspect._empty:
                    annotation = None
                function_parameters[key] = self._make_impl(annotation, parameters[key], **kwargs)
                print(f"[_make_from_function] Add argument {key} with value {function_arguments[key]}")
            elif key in kwargs.keys():
                function_parameters[key] = kwargs[key]
            elif value.default is not inspect._empty:
                pass
            else:
                raise ValueError(f"{key} is not contained in parameters and kwargs")
        return function(**function_parameters)

    def iterative_make(self, order, parameters, **kwargs):
        new_element = None
        for name in order:
            class_name = self._get_class_name(parameters[name], name)
            new_element = self.make(self._classes[class_name], parameters[name], **kwargs)
            kwargs[name] = new_element
        return new_element

    @staticmethod
    def _get_class_name(parameters, name):
        if "name" in parameters.keys():
            return parameters["name"]
        elif "type" in parameters.keys():
            return parameters["type"]
        raise ValueError(f"Parameters for {name} don't contain name of class")
