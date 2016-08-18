#!/bin/bash
echo 'Ejecutando: install_utilities.sh'
if which vim&>/dev/null; then
  echo 'Utilities ya están instalados. Nada que hacer.'
else
  sudo yum install -y vim
  sudo yum install -y git
  sudo yum install -y nmap
  sudo yum install -y tree
  #sudo yum install -y w3m
fi
