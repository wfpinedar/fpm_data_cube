#!/bin/bash
# Install Deps
sudo yum install -y epel-release
sudo yum localinstall -y http://yum.postgresql.org/9.3/redhat/rhel-7-x86_64/pgdg-centos93-9.3-1.noarch.rpm
sudo yum update -y
sudo yum install -y postgresql93-server postgresql93 postgresql93-contrib postgresql93-libs PyQt4 PyQt4-devel PyQt4-webkit gdal gdal-devel gdal-libs gdal-python numpy numpy-f2py postgis2_93 postgis2_93-client postgis2_93-debuginfo postgis2_93-devel postgis2_93-docs postgis2_93-utils python-psycopg2 pytz scipy pgadmin3_93 environment-modules wget git gcc gcc-c++ unzip
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install numexpr
sudo pip install ephem
git clone https://github.com/GeoscienceAustralia/EO_tools.git --depth=10
# Install EOtools
cd EO_tools/
sudo python setup.py install
cd ..
# Download and Install Data Cube
# Only get the develop branch
git clone -b develop --single-branch https://github.com/GeoscienceAustralia/agdc.git
# Change agdc_default.config
vim agdc/agdc/agdc_default.config
cp agdc/agdc/agdc-example.conf agdc/agdc/agdc_default.config
sed -i.bak 's/host\ \=\ 130\.56\.244\.224/host\ \=\ 0\.0\.0\.0/g' agdc/agdc/agdc_default.config
sed -i 's/port = 6432/port = 5432/g' agdc/agdc/agdc_default.config
sed -i 's/dbname = datacube/dbname = postgres/g' agdc/agdc/agdc_default.config
# Install Data Cube
cd agdc
sudo python setup.py install --force
# Necessary Modifications
# Update config file for API
sed -i.bak 's/HOST = "host"/HOST = "0.0.0.0"/g' api/source/main/python/datacube/config.py
sed -i 's/PORT = "port"/PORT = "3128"/g' api/source/main/python/datacube/config.py
sed -i 's/DATABASE = "database"/DATABASE = "postgres"/g' api/source/main/python/datacube/config.py
sudo /usr/pgsql-9.3/bin/postgresql93-setup initdb
sudo systemctl enable postgresql-9.3.service
sudo systemctl start postgresql-9.3.service
# Create database roles
tee /tmp/create_databases_roles.sql << EOF
create user cube_admin with superuser;
create user cube_user with superuser;
create user jeremyhooke with superuser;
create group cube_user_group;
create group cube_admin_group with superuser;
\q
EOF
sudo chown postgres:postgres /tmp/create_databases_roles.sql
sudo su postgres -c "
psql -f /tmp/create_databases_roles.sql
"
# Change permissions
#vim /var/lib/pgsql/9.3/data/pg_hba.conf
#Towards the bottom, the second local host should be trust
#Hacia la parte inferior, el segundo anfitrión local debe ser de confianza.
#exit
