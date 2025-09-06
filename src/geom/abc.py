import abc


class BaseFigure2D(metaclass=abc.ABCMeta):
    """ Abstract class of 2D figure """

    @abc.abstractmethod
    def area(self):
        pass


class BaseShape2D(metaclass=abc.ABCMeta):
    """ Abstract class of 2D shape """

    @abc.abstractmethod
    def area(self):
        pass
