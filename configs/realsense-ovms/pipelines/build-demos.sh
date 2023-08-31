#!/bin/bash

for fld in $(find . -maxdepth 1 -type d)
do
  if [ $fld != "." ]; then
    cd $fld; ls -l; make || true;
    cd ..
  fi
  
done