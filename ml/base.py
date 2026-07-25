from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all models, standardizing fit and predict methods (implementation is still model-specific)
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X, y=None):
        """
        Train the model on X(and y if supervised). Must set self._is_fitted = True when done.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """
        Generate predictions for new data X.

        Must raise an error if called before fit().
        """
        raise NotImplementedError

    def _check_is_fitted(self):
        """
        Shared helper: raises an error if predict() is called before fit().
        Subclasses call this at the top of their predict() method.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling predict()."
            )