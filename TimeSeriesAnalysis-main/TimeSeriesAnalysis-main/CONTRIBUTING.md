# Contributing to this repo
In order to keep the repository tidy, we should follow a predetermined workflow and try our best to adhere to a style guide.

## Git Workflow
We basically use the
[Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
as our basic git workflow. A brief summary:

* We always create a branch from the master branch when modifying the code.
Branches are named something like `ad-hoc-branch-name` or `ad-hoc-branch-name-issue-XXX` if the fix corresponds to a certain issue.  
* We integrate code by making pull requests. Pull requests are reviewed by one or more other developers. 
* We delete branches after merging the pull request.
* By default, _branches are personal_. That means that we do not produce commits on branches created by other uses
(unless after agreement). As a corollary, a user can rewrite history by rebasing a branch or squashing commits without
first notifying others.

## Code style 
We adhere to the [PEP-8](https://www.python.org/dev/peps/pep-0008/) style guide with the following
exceptions/clarifications:
* Maximum line length is 120 characters

Docstrings should be provided to all non-trivial modules, classes and functions/methods.
We use [google style](http://google.github.io/styleguide/pyguide.html) guidelines for docstrings (see also the
 following concise [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

We use [PEP-484](https://www.python.org/dev/peps/pep-0484/) type hints.

We use _english_ in all code comments and when writing/commenting pull requests. 