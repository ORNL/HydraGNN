# Contributing

Contributing to HydraGNN is easy: just open a [pull
request](https://help.github.com/articles/using-pull-requests/). Make
`master` the destination branch on the [HydraGNN
repository](https://github.com/ORNL/HydraGNN) and allow edits from
maintainers in the pull request.

Your pull request must pass HydraGNN's tests, be formatted to match HydraGNN, and be
reviewed by at least one HydraGNN developer.

Additional dependencies are needed for formatting and testing:
```
pip install -r requirements-dev.txt
```

## Code formatting

HydraGNN is formatted with `black`. You should run `black .` from the top level
directory.

## Unit testing

HydraGNN uses `pytest` to test. You can run `python -m pytest` from the top level
directory.
