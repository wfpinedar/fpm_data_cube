#!/bin/bash
echo 'Ejecutando: set_permisive.sh'
if egrep '^SELINUX=permissive' /etc/selinux/config &>/dev/null; then
  echo 'SELINUX permisivo y FIREWALLD desactivado. Nada que hacer.'
else
# FIREWALLD
SUDO=sudo
$SUDO systemctl disable firewalld
$SUDO systemctl stop firewalld

# SELINUX
# rationale: TODO
echo Configurando SELINUX
if type setenforce &>/dev/null && [ "$(getenforce)" != "Disabled" ]
then
  echo setenforce permissive
  $SUDO setenforce permissive
fi
if [ -f /etc/selinux/config ]
then
  $SUDO sed -i.bak 's/^SELINUX=.*/SELINUX=permissive/' /etc/selinux/config
  egrep '^SELINUX=' /etc/selinux/config
fi
fi
