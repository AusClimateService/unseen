## Contributor guide

This package is still is its very early stages of development. The following covers some general guidelines for maintainers and contributors.

#### Preparing Pull Requests
 1. For this respository. It's fine to use `unseen` as your fork repository name because it will live under your username.

 2. Clone your fork locally, connect your repository to the upstream (main project), and create a branch to work on:

```
$ git clone git@github.com:YOUR_GITHUB_USERNAME/unseen.git
$ cd unseen
$ git remote add upstream git@github.com:AusClimateService/unseen.git
$ git checkout -b your-bugfix-feature-branch-name master
```

 3. Install `unseen`'s dependencies into a new conda environment:

```
$ conda env create -f environment.yml
$ conda activate unseen
```

 4. Install `unseen` using the editable flag (meaning any changes you make to the package will be reflected directly in your environment):

```
$ pip install --no-deps -e .
```

 5. Start making and committing your edits, including adding tests to `unseen/tests` to check that your contributions are doing what they're suppose to. To run the test suite:

```
pytest unseen
```