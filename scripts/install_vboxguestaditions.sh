#!/bin/bash
sudo su -c "
yum –exclude=kernel* update -y
yum install -y gcc
yum install -y kernel-devel-$(uname -r)
yum install -y wget
echo 'Descargando VBox image'
wget -c http://download.virtualbox.org/virtualbox/5.0.20/VBoxGuestAdditions_5.0.20.iso -O VBoxGuestAdditions.iso > /dev/null 2>&1
echo 'Descarga Terminada'
mount VBoxGuestAdditions.iso -o loop /mnt
/mnt/VBoxLinuxAdditions.run
#rm -rf VBoxGuestAdditions.iso
"
