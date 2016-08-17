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
  $SUDO sed -i.packer-bak 's/^SELINUX=.*/SELINUX=permissive/' /etc/selinux/config
  egrep '^SELINUX=' /etc/selinux/config
fi

