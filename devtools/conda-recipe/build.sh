#!/bin/bash

cp -r $RECIPE_DIR/../.. $SRC_DIR
$PYTHON setup.py clean
$PYTHON setup.py install

# Push examples to anaconda/share/bhmm/examples/
#mkdir $PREFIX/share/bhmm
#cp -r $RECIPE_DIR/../../examples $PREFIX/share/bhmm/
