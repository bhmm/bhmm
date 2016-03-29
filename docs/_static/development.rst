Development notes
=================

Releases
--------
**Checklist**

    * Do all tests pass?
    * Are the changes tested in the upstream/dependent software?
    * Get twine: ```pip install twine```

**Steps**

    * Create a new tag with git on your local working copy:

    ```git tag 1.0 -m "Release message"```

    * Push the tag to origin:

    ```git push --tags```

    * Update the conda recipe for Omnia channel

        * Create a fork of https://github.com/omnia-md/conda-recipes
        * Edit the recipe (meta.yml) to match the new dependencies and version numbers.
        * Create a pull request on https://github.com/omnia-md/conda-recipes/pulls to have your changes tested and merged.
        * Sit back and wait for the packages being built, merge when succeeding.

    * PyPI release:

        * Create a source archive $name-$version.tar.gz file in the dist directory:

        ```python setup.py sdist```

        * Upload to PyPI:

        ```twine upload dist/$name-$version.tar.gz --user $user --password $pass```


