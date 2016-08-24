Datacube v1.5-NASA

Ambiente:
	RAM: 64GB
	DISCO: 1TB
	CORES: 20
	Sistema Operativo: Ubuntu 14.04
Instalación
1. descomprimir datacube-working.tgz
2. ejecutar DataCube-dist/installer/dc_install.sh
3. copiar los datos ingestados a /tilestore/data2
4. escribir el archivo kenya_cells.txt en /tilestore, usando for a in $(seq -180 180); do for b in $(seq -90 90); do echo "$a" "$b"; done; done > kenya_cells.txt
5. restaurar el backup de la base de datos. sudo -u postgres psql < colombia.sql
a. por alguna razón la primera vez que se realizó no restauró algunos datos (por lo que fallaban las pruebas), se corrige volviendo a importar el script.
6. usar pip para instalar celery, flask, redis y amqp sudo pip install celery flask flask-cors redis amqp pyrabbit
7. instalar rabittmq y redis server sudo apt-get install rabbitmq-server redis-server
8. iniciar celery usando el archivo datacube_worker.py:  celery worker -A datacube_worker --purge
9. Crear el directorio /tilestore/tile_cache
10. probar:
a. Editar el archivo make_image.py con coordenadas existentes y ejecutarlo



Cosas que aprendimos:
Esta versión paraleliza, usando Celery.
La arquitectura tiende a converger con la versión 2.
Tiene un ingester válido para Landsat 7 (USGS)

#Para los cambios que se realizaron se necesita numpy ==1.9 sklearn
sudo pip install sklearn numpy==1.9
sudo apt-get install libfreetype6 libfreetype6-dev
sudo -H pip install matplotlib

Para que funcione parte de la interfaz se requiere imagemagick

sudo apt-get install imagemagick apache2
