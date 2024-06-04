from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    assert arg < len(vals)
    vals1 = list(vals)  # vals is seen as a Tuple by default
    vals1[arg] += epsilon
    val1: float = f(*vals1)

    vals2 = list(vals)
    vals2[arg] -= epsilon
    val2: float = f(*vals2)

    return (val1 - val2) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited: List[int] = []
    reverse_order: List[Variable] = []

    def _dfs(scalar: Variable) -> None:
        visited.append(scalar.unique_id)
        if (
            not scalar.is_constant()
        ):  # assert scalar.history is not None, so scalar.parents is not None, aka pruning
            for var in scalar.parents:
                if var.unique_id not in visited:
                    _dfs(var)

        reverse_order.append(scalar)

    _dfs(variable)
    reverse_order.reverse()  # return None

    return reverse_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topo: Iterable[Variable] = topological_sort(variable)
    intermediate: Dict[int, float] = {
        var.unique_id: 0 for var in topo
    }  # The key of dict should be immutable, so we can't use Scalar.
    intermediate[variable.unique_id] = deriv

    for var in topo:
        if var.is_leaf():
            var.accumulate_derivative(intermediate[var.unique_id])
            continue

        if var.is_constant():  # chain_rule() must called on a non-constant Variable
            continue

        ls = var.chain_rule(intermediate[var.unique_id])
        for var_in, deriv in ls:
            intermediate[var_in.unique_id] += deriv  # ls: Tuple[Variable, float]


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
