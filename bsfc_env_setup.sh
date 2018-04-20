#!/bin/bash 

cwd=$PWD

cd /tmp
rm -rf bsfc_env
mkdir bsfc_env_tmp
cd bsfc_env_tmp
tar xfz $cwd'/bsfc_env.tgz'
mv bsfc_env ..
cd ..
rm -r bsfc_env_tmp
cd $cwd
