from typing import List, Any, Iterable, TypeVar, overload, Union
import collections.abc

_T = TypeVar("_T")
_SmarterSet = TypeVar("_SmarterSet")

class SmarterSet(set):
    """
    Same as a regular python set but the set will return itself when using add() or clear().
    """
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, __iterable: Iterable[_T]) -> None:
        ...

    def __str__(self) -> str:
        ...

    def add(self, __element: Any) -> _SmarterSet:
        ...

    def clear(self) -> _SmarterSet:
        ...

    def __init__(self, __arg1: Union[Iterable[_T], None] = None) -> None:
        if __arg1 == None:
            super().__init__()
        elif isinstance(__arg1, collections.abc.Iterable):
            super().__init__(__arg1)
        else:
            raise TypeError("Cannot create a SmarterSet from a non-iterable.")

    def __str__(self) -> str:
        # TODO theres probably a better and faster way to do this
        return "{" + ", ".join([f"{e}" for e in self]) + "}"

    def add(self, __element: Any) -> _SmarterSet:
        '''
        Add an element to a set and return the set.

        This has no effect if the element is already present.
        '''
        super().add(__element)
        return self

    def clear(self) -> _SmarterSet:
        '''
        Clear the set and return the set.
        '''
        super().clear()
        return self