# Documentation

The package comes with a documentation that can be generated using Sphinx. 

## Creating HTML documentation

```shell
cd doc/
make html
```

Other formats for the documentation are available as well; see the [Sphinx documentation](https://www.sphinx-doc.org/en/master/) for further details.


## Updating the documentation

If you add a new module, you need to run the following command within the docs folder before make:

```
sphinx-apidoc -eMf -o source ../src/adaptive_stratification
```
