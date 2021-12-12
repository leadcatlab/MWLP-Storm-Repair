# from graph import graph
# import algos
from typing_extensions import TypedDict
import testing

graphDict = TypedDict(
    "graphDict",
    {
        "numNodes": int,
        "nodeWeight": list[int],
        "edges": list[tuple[int, int, float]],
    },
)


def main() -> None:
    testing.benchmark(8, 200, metric=True, upper=5)


if __name__ == "__main__":
    main()
