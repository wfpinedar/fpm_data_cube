#!/bin/bash

# rationale: validar que es Centos
if ! cat /etc/os-release | grep -i 'centos' &> /dev/null
then
  echo 'Su distribución no es Centos.'
  exit 1
fi

# rationale: anunciar que si no tiene Centos versión 7 el instalador puede fallar
if ! cat /etc/os-release | grep -i 'VERSION="7"' &> /dev/null
then
  echo 'Su distribución no es Centos versión 7, tal vez no funcione.'
fi

# rationale: utilidades de bash para realizar scripts
file=/tmp/utils
if [ ! -f $file ]
then
  echo 'Descargando Utilidad Scripts BASH.'
  curl https://raw.githubusercontent.com/GLUD/bash_utils/master/utils -o $file
fi

# rationale: se importan utilidades
source $file

# rationale: Mensaje de bienvenida al isntalador
echo;echo
e_header 'Bienvenido al instalador del cubo de datos. author: @wfpinedar'
e_note 'Instalando utilidades necesarias para el script...'

# rationale: Se actualiza como recomienda la documentación de Red Hat
# link: https://access.redhat.com/articles/11258
sudo yum update -y

# rationale: epel-release: instala el repositorio community de Centos
sudo yum install -y epel-release

# rationale: figlet: muestra letreros vistosos en terminal, necesita del paquete epel-release
# rationale: wget: se necesita para descargar archivos de internet
sudo yum install -y figlet wget

# rationale: repositorio de postgresql oficial
# link: https://www.postgresql.org/download/linux/redhat/
sudo yum localinstall -y http://yum.postgresql.org/9.3/redhat/rhel-7-x86_64/pgdg-centos93-9.3-1.noarch.rpm

# rationale: mensaje Instalando dependencias del Sistema Operativo
figlet -f big Instalando Dependencias

# rationale: postgresql93-server: Instalación de PostgreSQL
# rationale: postgresql93: Instalación de PostgreSQL
# rationale: postgresql93-contrib: Instalación de PostgreSQL
# rationale: postgresql93-libs: Instalación de PostgreSQL
# rationale: PyQt4:
# rationale: PyQt4-devel:
# rationale: PyQt4-webkit:
# rationale: gdal:
# rationale: gdal-devel:
# rationale: gdal-libs:
# rationale: gdal-python:
# rationale: numpy:
# rationale: numpy-f2py:
# rationale: postgis2_93:
# rationale: postgis2_93-client:
# rationale: postgis2_93-debuginfo:
# rationale: postgis2_93-devel:
# rationale: postgis2_93-docs:
# rationale: postgis2_93-utils:
# rationale: python-psycopg2:
# rationale: pytz:
# rationale: scipy:
# rationale: pgadmin3_93: Administrador gráfico de bases de datos ¿No necesario?
# rationale: environment-modules:
# rationale: wget: Descargador de recursos de red
# rationale: git: Gestor de versionamiento de código
# rationale: gcc: Compiladores de código C, C++ y otros
# rationale: gcc-c++: Compiladores de código C, C++ y otros
# rationale: unzip: Descompresor de archivos ZIP
# rationale: gcc-gfortran: Compilador fortran
# rationale: rabbitmq-server:
# rationale: redis:
# rationale: tkinter:
# rationale: freetype-devel:
# rationale: ImageMagick:
# rationale: httpd:
sudo yum install -y postgresql93-server postgresql93 postgresql93-contrib postgresql93-libs ̣\
  PyQt4 PyQt4-devel PyQt4-webkit gdal gdal-devel gdal-libs gdal-python numpy numpy-f2py postgis2_93 \
  postgis2_93-client postgis2_93-debuginfo postgis2_93-devel postgis2_93-docs postgis2_93-utils \
  python-psycopg2 pytz scipy pgadmin3_93 environment-modules wget git gcc gcc-c++ unzip gcc-gfortran \
  rabbitmq-server redis tkinter freetype-devel ImageMagick httpd

# rationale: Instalando Python Packaget Index(pip): Gestor recomendado para instalar paquetes
# link: https://packaging.python.org/current/
if pip --version &> /dev/null
then
  e_success 'PIP ya está instalado. Nada que hacer.'
else
  figlet -f big 'Instalando PIP'
  e_warning 'wget https://bootstrap.pypa.io/get-pip.py'
  wget https://bootstrap.pypa.io/get-pip.py
  e_warning 'sudo python get-pip.py'
  sudo python get-pip.py
  e_note 'Terminado.'
fi #pip

# rationale: Mensaje Instalación de paquetes de python
figlet -f big 'Instalando paquetes python'

# rationale: numexpr:
# rationale: ephem:
# rationale: celery:
# rationale: flask:
# rationale: flask-cors:
# rationale: redis:
# rationale: amqp:
# rationale: pyrabbit:
# rationale: sklearn:
# rationale: numpy==1.9:
sudo pip install numexpr ephem celery flask flask-cors redis amqp pyrabbit enum matplotlib Image \
  sklearn numpy==1.9

# rationale: Instalando Earth Observation Tools
# rationale: se valida si EOTools está instalado
if python <<< 'import eotools' &> /dev/null
then
  e_success 'Earth Observation Tools ya está instalado. Nada que hacer.'
else
  figlet -f big 'Instalando Earth Observation Tools'
  # Descarga del repositorio de Earth Observation Tools
  git clone https://github.com/GeoscienceAustralia/EO_tools.git -b stable --single-branch --depth=10
  # Instalan en python EO_tools
  e_warning 'Instalando módulo Earth Observation Tools'
  cd EO_tools/
  sudo python setup.py install
  cd ..
  e_note 'Terminado.'
fi #eotools

# rationale: Instalando Australian Geoscience Data Cube (AGDC)
# rationale: se valida si AGDC está instalado
if python <<< 'import agdc' &> /dev/null
then
  e_success 'Australian Geoscience Data Cube (AGDC) ya está instalado. Nada que hacer.'
else
  figlet -f big 'Instalando Australian Geoscience Data Cube (AGDC)'
  git clone -b develop --single-branch https://github.com/GeoscienceAustralia/agdc.git
  # Change agdc_default.config
  cp agdc/agdc/agdc-example.conf agdc/agdc/agdc_default.config
  sed -i.bak 's/host\ \=\ 130\.56\.244\.224/host\ \=\ 0\.0\.0\.0/g' agdc/agdc/agdc_default.config
  sed -i 's/port = 6432/port = 5432/g' agdc/agdc/agdc_default.config
  sed -i 's/dbname = datacube/dbname = postgres/g' agdc/agdc/agdc_default.config
  # Install Data Cube
  cd agdc
  e_header 'Instalando módulo AGDC'
  sudo python setup.py install --force
  # Necessary Modifications
  # Update config file for API
  sed -i.bak 's/HOST = "host"/HOST = "0.0.0.0"/g' api/source/main/python/datacube/config.py
  sed -i 's/PORT = "port"/PORT = "3128"/g' api/source/main/python/datacube/config.py
  sed -i 's/DATABASE = "database"/DATABASE = "postgres"/g' api/source/main/python/datacube/config.py
  cd ..
  e_note 'Terminado.'
fi #agdc

# rationale: Realizando configuración POSTGRESQL
# rationale: si el archivo $file existe, es porque postgresql ya está inicializado
file=/var/lib/pgsql/9.3/data/pg_hba.conf
if [ -f $file ]
then
  e_success 'PostgreSQL ya está configurado. Nada que hacer.'
else
  figlet -f big 'Inicializando PostgreSQL'
  sudo /usr/pgsql-9.3/bin/postgresql93-setup initdb
  sudo systemctl enable postgresql-9.3.service
  sudo systemctl start postgresql-9.3.service
  e_note 'Terminado.'
fi #/var/lib/pgsql/9.3/data/pg_hba.conf

# Change permissions
#Towards the bottom, the second local host should be trust
#Hacia la parte inferior, el segundo anfitrión local debe ser de confianza.
file=/var/lib/pgsql/9.3/data/pg_hba.conf
if [ -f $file'.bak' ]
then
  e_success $file'.bak existe. Nada que hacer.'
else
  sed -i.bak '/^local/ s/peer/trust/' $file
  #sed -i '/^host/ s/ident/md5/' /var/lib/pgsql/9.3/data/pg_hba.conf
  #echo 'host    all             all             0.0.0.0/0               md5' >> /var/lib/pgsql/9.4/data/pg_hba.conf
  sudo systemctl restart postgresql-9.3.service
fi #/var/lib/pgsql/9.3/data/pg_hba.conf.bak

# rationale: crear roles en base de datos
# rationale: se valida si existe el script, si no, se crea y se ejecuta
file=/tmp/create_databases_roles.sql
if [ -f $file ]
then
  e_success $file' existe y está ejecutado. Nada que hacer.'
else
  figlet -f big 'Creando roles base de datos'
  # tal vaz falte un ALTER USER usuario WITH PASSWORD 'clave'
  tee /tmp/create_databases_roles.sql << EOF
  CREATE USER cube_admin WITH superuser;
  CREATE USER cube_user WITH superuser;
  CREATE USER jeremyhooke WITH superuser;
  CREATE GROUP cube_user_group;
  CREATE GROUP cube_admin_group WITH superuser;
  \q
  EOF
  sudo chown postgres:postgres /tmp/create_databases_roles.sql
  sudo -u postgres psql -f /tmp/create_databases_roles.sql
  e_note 'Terminado.'
fi #/tmp/create_databases_roles.sql

#rationale
path=scripts_sql
if [ -d $path ]
then
  e_success 'scripts_sql existe. Nada que hacer.'
else
  figlet -f big 'Ejecutando Scripts SQL'
  mkdir $path
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
  e_note 'Terminado.'
fi #scripts_sql
