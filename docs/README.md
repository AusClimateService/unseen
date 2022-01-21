# Documentation

Detailed documentation for the UNSEEN package can be generated as follows:

```bash
$ cd unseen/docs
$ make html
```

After running `make html` you can view the documentation in your web browser
by opening the following file:

```
unseen/docs/_build/html/index.html
```

If you'd rather not go to the trouble of generating the html pages,
the reStructuredText (`.rst`) files in the `getting_started/` and `user_guide/` directories
contain the original information used to build the html pages.


## Required software

In order to build the html pages,
the UNSEEN package and all its dependencies need to be installed in the environment
you're working from
(see [installation instructions](https://github.com/AusClimateService/unseen/blob/master/docs/getting_started/index.rst)).

The following additional packages are also required specifically for the html build:

```bash
$ conda install sphinx numpydoc pydata-sphinx-theme
```

## Development notes

In order to create an API reference where you can click through to function definitions,
we needed to create a couple of custom template files (see `_templates/`) following 
[this Stack Overflow comment](https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough/62613202#62613202).
