#!/bin/bash
if [ ! -f /tmp/utils ]
then
  curl https://raw.githubusercontent.com/GLUD/bash_utils/master/utils -o /tmp/utils
fi
. /tmp/utils
e_header "\n\n"Bienvenido al instalador de cubo de datos del Fondo Patrimonio Natural. author: @wfpinedar
e_note 'Instalando utilidades necesarias para el script...'
# rationale: epel-release: instala el repositorio community de Centos
# rationale: figlet: muestra letreros vistosos en terminal
sudo yum install -y epel-release
sudo yum install -y figlet

# Install Deps
figlet -f big Instalando Dependencias
# Se actualiza como recomienda la documentación de Red Hat
# link: https://access.redhat.com/articles/11258
#sudo yum update -y #No va de primero?
# Repositorio de postgresql oficial
# link: https://www.postgresql.org/download/linux/redhat/
sudo yum localinstall -y http://yum.postgresql.org/9.3/redhat/rhel-7-x86_64/pgdg-centos93-9.3-1.noarch.rpm
# rationale: postgresql93-server postgresql93 postgresql93-contrib postgresql93-libs: Instalación de PostgreSQL
# rationale: PyQt4 PyQt4-devel PyQt4-webkit: 
# rationale: gdal gdal-devel gdal-libs gdal-python:
# rationale: numpy numpy-f2py:
# rationale: postgis2_93 postgis2_93-client postgis2_93-debuginfo postgis2_93-devel postgis2_93-docs postgis2_93-utils:
# rationale: python-psycopg2:
# rationale: pytz:
# rationale: scipy:
# rationale: pgadmin3_93: Administrador gráfico de bases de datos ¿No necesario?
# rationale: environment-modules:
# rationale: wget: Descargador de recursos de red
# rationale: git: Gestor de versionamiento de código
# rationale: gcc gcc-c++: Compiladores de código C, C++ y otros
# rationale: unzip: Descompresor de archivos ZIP
# rationale: gcc-gfortran: Compilador fortran 
sudo yum install -y postgresql93-server postgresql93 postgresql93-contrib postgresql93-libs PyQt4 PyQt4-devel PyQt4-webkit gdal gdal-devel gdal-libs gdal-python numpy numpy-f2py postgis2_93 postgis2_93-client postgis2_93-debuginfo postgis2_93-devel postgis2_93-docs postgis2_93-utils python-psycopg2 pytz scipy pgadmin3_93 environment-modules wget git gcc gcc-c++ unzip gcc-gfortran

# rationale: pip: Python Packaget Index: Gestor recomendado para instalar paquetes
# link: https://packaging.python.org/current/
if [ -f get-pip.py ]
then
  e_success 'El archivo get-pip.py ya está descargado y ejecutado. Nada que hacer.'
else

e_warning 'wget https://bootstrap.pypa.io/get-pip.py'
wget https://bootstrap.pypa.io/get-pip.py
e_warning 'sudo python get-pip.py'
sudo python get-pip.py

fi #get-pip.py

# Instalación de paquetes de python
e_warning 'sudo pip install numexpr ephem'
sudo pip install numexpr ephem

# Install EOtools
# Descarga del repositorio de EOtools
if [ -d EO_tools ]
then
  e_success 'EO_tools ya está instalado. Nada que hacer.'
else 

git clone https://github.com/GeoscienceAustralia/EO_tools.git -b stable --single-branch --depth=10
figlet -f big Instalando EOtools
cd EO_tools/
# Instalan dependencias de EO_tools
e_warning 'sudo pip install numexpr ephem'
sudo python setup.py install
cd ..

fi #EO_tools

# Download and Install Data Cube
figlet -f big Instalando Data Cube
# Only get the develop branch
if [ -d agdc ]
then
  e_success 'agdc ya está instalado. Nada que hacer.'
else

git clone -b develop --single-branch https://github.com/GeoscienceAustralia/agdc.git
# Change agdc_default.config
cp agdc/agdc/agdc-example.conf agdc/agdc/agdc_default.config
sed -i.bak 's/host\ \=\ 130\.56\.244\.224/host\ \=\ 0\.0\.0\.0/g' agdc/agdc/agdc_default.config
sed -i 's/port = 6432/port = 5432/g' agdc/agdc/agdc_default.config
sed -i 's/dbname = datacube/dbname = postgres/g' agdc/agdc/agdc_default.config
# Install Data Cube
cd agdc
e_header 'Instalando dependencias AGDC'
sudo python setup.py install --force
# Necessary Modifications
# Update config file for API
sed -i.bak 's/HOST = "host"/HOST = "0.0.0.0"/g' api/source/main/python/datacube/config.py
sed -i 's/PORT = "port"/PORT = "3128"/g' api/source/main/python/datacube/config.py
sed -i 's/DATABASE = "database"/DATABASE = "postgres"/g' api/source/main/python/datacube/config.py
cd ..

fi #agdc

# Realizando configuración POSTGRESQL
if [ -f /var/lib/pgsql/9.3/data/pg_hba.conf ]
then
  e_success 'PostgreSQL ya está configurado. Nada que hacer.'
else

sudo /usr/pgsql-9.3/bin/postgresql93-setup initdb
sudo systemctl enable postgresql-9.3.service
sudo systemctl start postgresql-9.3.service

fi #/var/lib/pgsql/9.3/data/pg_hba.conf

# Change permissions
#Towards the bottom, the second local host should be trust
#Hacia la parte inferior, el segundo anfitrión local debe ser de confianza.
if [ -f /var/lib/pgsql/9.3/data/pg_hba.conf.bak ]
then
  e_success '/var/lib/pgsql/9.3/data/pg_hba.conf.bak existe. Nada que hacer.'
else

sed -i.bak '/^local/ s/peer/trust/' /var/lib/pgsql/9.3/data/pg_hba.conf
sudo systemctl restart postgresql-9.3.service

fi #/var/lib/pgsql/9.3/data/pg_hba.conf.bak

# Create database roles
if [ -f /tmp/create_databases_roles.sql ]
then
  e_success '/tmp/create_databases_roles.sql existe y está ejecutado. Nada que hacer.'
else

e_note 'Creando roles base de datos'
tee /tmp/create_databases_roles.sql << EOF
create user cube_admin with superuser;
create user cube_user with superuser;
create user jeremyhooke with superuser;
create group cube_user_group;
create group cube_admin_group with superuser;
\q
EOF
sudo chown postgres:postgres /tmp/create_databases_roles.sql
sudo -u postgres psql -f /tmp/create_databases_roles.sql

fi #/tmp/create_databases_roles.sql

if [ -d scripts_sql ]
then
  e_success 'scripts_sql existe. Nada que hacer.'
else

figlet -f big Ejecutando Scripts SQL
mkdir scripts_sql
cd $_
wget https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/scripts_sql/v1__schema.sql
wget https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/scripts_sql/v2__base_data.sql
wget https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/scripts_sql/v3__additional_constraints.sql
wget https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/scripts_sql/v4__modis.sql
wget https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/scripts_sql/v5__wofs.sql
wget https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/scripts_sql/v6__index.sql
for i in $(ls)
do
  e_warning "Ejecutando: $i"
  sudo -u postgres psql < $i
done
cd ..

fi #scripts_sql

#Install API
#sudo python setup.py install –force
# Create directory to store tiles
if [ -d /g ]
then
  e_success '/g existe. Nada que hacer.'
else

sudo mkdir /g/
chmod 777 /g/
e_success 'Directirio /g/ generado'

fi # /g
