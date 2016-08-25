#Siguiendo el manual create_datacube
Es el entorno de desarrollo.

Se ejecuta la provisión:
------------------------
```bash
$ vagrant up
$ vagrant provision
$ vagrant reload
$ vagrant ssh
```

¿MÁS FÁCIL?
-----------
```bash
$ ./install
```

INICIAR
-------
```bash
$ ./run
```

REINICIAR
---------
```bash
$ ./run r
```

Con este trabajo se pretende:
-----------------------------

* La instalación del cubo de datos puede llegar a desplegarse en cualquier ordenador por medio de un entorno de virtualización con vagrant en cuestión de minutos.
* Puede también ser realizado por cualqiuera sin mayores conocimientos sobre sistemas GNU/Linux y en la comodidad de su sistema operativo Windows, MAC o Linux.
* Documenta la inclusión de paquetes para replicabilidad del entorno y resolución de dependencias para otras distribuciones, mejorando así el misticismo de paquetes y sus dependencias.
* Normaliza las versiones de paquetes, módulos y sus dependencias que se deben instalar dentro de la distribución Centos 7.
* Se centraliza la información de instalación ya que en un principio los ficheros/archivos requeridos estaban dispersos en los correos de los relacionados al proyecto.
* Tener un versionamiento de código con git permite el control y escalamiento del proyecto de generación de imagen de centos 7 con el cubo instalado que se ha llevado a cabo en estas semanas. https://git-scm.com/book/es/v1/Empezando-Acerca-del-control-de-versiones
* Por lo tanto con este trabajo se ha generado facilidad para desplegar entornos de desarrollo, pruebas y producción como se recomienda para todo desarrollo de software. https://styde.net/tipos-de-servidores-y-entornos

