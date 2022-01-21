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

The following packages need to be installed in order to generate the html documentation:

```bash
$ conda install sphinx numpydoc pydata-sphinx-theme
```

## Development notes

In order to create an API reference where you can click through to function definitions,
we needed to create a couple of custom template files (see `_templates/`) following 
[this Stack Overflow comment](https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough/62613202#62613202).
