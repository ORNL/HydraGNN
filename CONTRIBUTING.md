# Contributing

Contributing to GCNN is easy: just open a [pull
request](https://help.github.com/articles/using-pull-requests/). Make
`master` the destination branch on the [GCNN
repository](https://github.com/allaffa/GCNN) and allow edits from
maintainers in the pull request.

Your pull request must pass GCNN's tests, be formatted to match GCNN, and be
reviewed by at least one GCNN developer.

Additional dependencies are needed for formatting and testing:
```
pip install -r requirements-dev.txt
```

## Code formatting

GCNN is formatted with `black`. You should run `black .` from the top level
directory.

## Unit testing

GCNN uses `pytest` to test. You can run `python -m pytest` from the top level
directory.
