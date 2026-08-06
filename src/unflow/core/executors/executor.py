from abc import ABC, abstractmethod


class Executor(ABC):
    @abstractmethod
    def submit(self, state):
        """Submit a state for execution."""
        pass

    @abstractmethod
    def wait(self):
        """Wait until one submitted job finishes."""
        pass
